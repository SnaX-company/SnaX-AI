# main.py - VERSÃO COM EARLY STOPPING E HIPERPARÂMETROS CONFIGURÁVEIS
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datasets import load_dataset
from model import SnaXIA
from tokenizer import SnaXTokenizer
import os
import numpy as np

# =============================================================================
# CONFIGURAÇÃO DE HIPERPARÂMETROS
# =============================================================================
CONFIG = {
    # --- Tokenizer & Data ---
    "vocab_size": 8000,
    "max_seq_length": 128,
    "test_set_size": 0.1,
    
    # --- Training ---
    "batch_size": 8,
    "learning_rate": 5e-5,
    "epochs": 10,
    "early_stopping_patience": 2,
    
    # --- Architecture ---
    "hidden_size": 128,
    "num_heads": 4,
    "dim_feedforward": 256,
    "num_layers": 2,
    "memory_slots": 16,
    "snn_beta": 0.9, # Parâmetro do neurônio Leaky
    
    # --- Saving ---
    "model_save_path": "snaxia_model_best.pth",
    "tokenizer_save_path": "tokenizer.model"
}

# =============================================================================
# 1. TOKENIZADOR
# =============================================================================
print("Verificando tokenizador...")
if not os.path.exists(CONFIG["tokenizer_save_path"]):
    print(f"Criando novo tokenizador com {CONFIG['vocab_size']} tokens...")
    tokenizer = SnaXTokenizer(vocab_size=CONFIG["vocab_size"])
else:
    print("Carregando tokenizador existente.")
    tokenizer = SnaXTokenizer(model_path=CONFIG["tokenizer_save_path"])

# Atualiza o vocab_size no config com o valor real do tokenizador
CONFIG["vocab_size"] = tokenizer.get_vocab_size()
print(f"Tamanho do vocabulário: {CONFIG['vocab_size']} tokens.")

# =============================================================================
# 2. MODELO
# =============================================================================
print("Inicializando o modelo SnaX IA com a nova configuração...")
model = SnaXIA(config=CONFIG) # Passa o dicionário de configuração inteiro
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)
print(f"Modelo movido para o dispositivo: {DEVICE}")

# =============================================================================
# 3. DATASET (COM DIVISÃO TREINO/VALIDAÇÃO)
# =============================================================================
print("Baixando e processando o dataset GSM8K...")
dataset = load_dataset("gsm8k", "main")["train"]

def preprocess(ex):
    q = ex["question"].strip()
    a = ex["answer"].split("####")[-1].strip()
    prompt = f"Q: {q}\nA:"
    # Truncar para garantir que não exceda o comprimento máximo
    input_ids = tokenizer.encode(prompt)[:CONFIG["max_seq_length"]]
    labels = tokenizer.encode(a)[:CONFIG["max_seq_length"]]
    return {"input_ids": input_ids, "labels": labels}

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
split_dataset = dataset.train_test_split(test_size=CONFIG["test_set_size"])
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(f"Dataset dividido: {len(train_dataset)} para treino, {len(val_dataset)} para validação.")

class SnaXDataset(Dataset):
    def __init__(self, dataset_split):
        self.dataset = dataset_split
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        item = self.dataset[i]
        return torch.tensor(item["input_ids"]), torch.tensor(item["labels"])

# Collate function para padding dinâmico
def collate_fn(batch):
    inputs, labels = zip(*batch)
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_id())
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return padded_inputs, padded_labels

train_loader = DataLoader(SnaXDataset(train_dataset), batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(SnaXDataset(val_dataset), batch_size=CONFIG["batch_size"], collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

# =============================================================================
# 4. TREINAMENTO COM EARLY STOPPING
# =============================================================================
print("Iniciando o treinamento com Early Stopping...")
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(CONFIG["epochs"]):
    # --- Fase de Treino ---
    model.train()
    total_train_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, CONFIG["vocab_size"]), y.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Grad clipping
        optimizer.step()
        
        total_train_loss += loss.item()
        if i % 100 == 0:
            print(f"Época {epoch+1}/{CONFIG['epochs']} | Passo {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)

    # --- Fase de Validação ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, CONFIG["vocab_size"]), y.view(-1), ignore_index=-100)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"\nÉPOCA {epoch+1} CONCLUÍDA")
    print(f"  -> Loss de Treino Média: {avg_train_loss:.4f}")
    print(f"  -> Loss de Validação Média: {avg_val_loss:.4f}")

    # --- Lógica do Early Stopping ---
    if avg_val_loss < best_val_loss:
        print("  -> Loss de validação melhorou! Salvando o modelo...")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), CONFIG["model_save_path"])
        tokenizer.save(CONFIG["tokenizer_save_path"])
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  -> Loss de validação não melhorou. Paciência: {patience_counter}/{CONFIG['early_stopping_patience']}")

    if patience_counter >= CONFIG['early_stopping_patience']:
        print("\nEarly stopping ativado! O modelo não está mais melhorando.")
        break
    print("-" * 30)

# =============================================================================
# 5. FIM
# =============================================================================
print("Treinamento concluído!")
print(f"O melhor modelo foi salvo em: '{CONFIG['model_save_path']}'")
print(f"O tokenizador foi salvo em: '{CONFIG['tokenizer_save_path']}'")
