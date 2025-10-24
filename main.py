# main.py
# Treinamento inicial da SnaX IA com GSM8K (raciocínio matemático)
# Roda 100% no Google Colab (GPU grátis)
# Salva: snaxia_model.pth + tokenizer.model

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datasets import load_dataset
from model import SnaXIA
from tokenizer import SnaXTokenizer
import os

# =============================
# CONFIGURAÇÕES
# =============================
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_SEQ_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"SnaX IA rodando em: {DEVICE}")

# =============================
# TOKENIZADOR
# =============================
print("Carregando SnaX Tokenizer...")
tokenizer = SnaXTokenizer()
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f"Vocabulário: {VOCAB_SIZE} tokens")

# =============================
# MODELO
# =============================
print("Inicializando SnaX IA...")
model = SnaXIA(vocab_size=VOCAB_SIZE, hidden_size=256, num_heads=4)
model = model.to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parâmetros: {total_params:,} (~{total_params//1_000_000}M)")

# =============================
# DATASET: GSM8K
# =============================
print("Baixando GSM8K (8 mil problemas de matemática)...")
dataset = load_dataset("gsm8k", "main")["train"]

def preprocess(example):
    """
    Formato:
    Q: Quanto é 15 + 27?
    A: 42
    """
    question = example["question"].strip()
    answer = example["answer"].split("####")[-1].strip()  # Pega só o número final
    prompt = f"Q: {question}\nA:"
    target = answer
    
    input_ids = tokenizer.encode(prompt)[:MAX_SEQ_LEN]
    label_ids = tokenizer.encode(target)[:MAX_SEQ_LEN]
    
    return {"input_ids": input_ids, "labels": label_ids}

print("Processando dataset...")
dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

class SnaXDataset(Dataset):
    def __len__(self): return len(dataset)
    def __getitem__(self, idx):
        item = dataset[idx]
        return (
            torch.tensor(item["input_ids"], dtype=torch.long),
            torch.tensor(item["labels"], dtype=torch.long)
        )

train_dataset = SnaXDataset()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =============================
# OTIMIZADOR
# =============================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
)

# =============================
# TREINAMENTO
# =============================
print(f"\nIniciando treinamento da SnaX IA por {EPOCHS} épocas...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for step, (input_ids, labels) in enumerate(train_loader):
        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
        ).to(DEVICE)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        ).to(DEVICE)
        
        # Forward
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if step % 50 == 0:
            print(f"Época {epoch+1}/{EPOCHS} | Passo {step} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"\nÉPOCA {epoch+1} CONCLUÍDA | Loss médio: {avg_loss:.4f}\n")

# =============================
# SALVAR
# =============================
print("Salvando SnaX IA...")
torch.save(model.state_dict(), "snaxia_model.pth")
tokenizer.save("tokenizer.model")

print("SnaX IA treinada com sucesso!")
print("Arquivos salvos:")
print("  → snaxia_model.pth")
print("  → tokenizer.model")

# =============================
# TESTE RÁPIDO
# =============================
def ask_snax(question):
    model.eval()
    prompt = f"Q: {question}\nA:"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(20):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            if next_token.item() == tokenizer.sp.piece_to_id("<|eot|>"):
                break
    
    return tokenizer.decode(input_ids[0].cpu().tolist())

print("\nTESTE DA SNAX IA:")
print(ask_snax("Quanto é 47 + 85?"))
print(ask_snax("Se um trem sai às 14h e chega às 17h30, quanto tempo durou a viagem?"))