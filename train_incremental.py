# train_incremental.py
# Treinamento incremental da SnaX IA
# Roda diariamente via GitHub Actions (grátis)
# - Atualiza tokenizador
# - Fine-tuning com novos dados
# - Salva modelo e tokenizador

import torch
import torch.nn.functional as F
from model import SnaXIA
from tokenizer import SnaXTokenizer
import os
import json
from datetime import datetime

print("SnaX IA: Iniciando treinamento incremental...")

# =============================
# CARREGAR MODELO E TOKENIZADOR
# =============================
if not os.path.exists("snaxia_model.pth"):
    print("ERRO: snaxia_model.pth não encontrado!")
    print("Rode main.py primeiro.")
    exit()

tokenizer = SnaXTokenizer()
VOCAB_SIZE = tokenizer.get_vocab_size()
model = SnaXIA(vocab_size=VOCAB_SIZE)
model.load_state_dict(torch.load("snaxia_model.pth", map_location="cpu"))
model.train()

print(f"Modelo carregado: {VOCAB_SIZE} tokens no vocabulário")

# =============================
# SIMULAR NOVOS DADOS (ou carregar de arquivo)
# =============================
# Em produção: leia de um arquivo JSON com interações do usuário
new_interactions = [
    {
        "question": "Quanto é 38 × 7?",
        "answer": "266"
    },
    {
        "question": "Se eu tenho 5 maçãs e dou 2, quantas ficam?",
        "answer": "3"
    },
    {
        "question": "Qual é a capital da França?",
        "answer": "Paris"
    },
    {
        "question": "SnaX IA é incrível!",
        "answer": "Obrigado! Estou aprendendo todo dia."
    }
]

# Extrai textos para atualizar tokenizador
new_texts = [q["question"] for q in new_interactions] + [q["answer"] for q in new_interactions]

# =============================
# ATUALIZAR TOKENIZADOR
# =============================
print("Atualizando tokenizador com novos textos...")
tokenizer.update_with_texts(new_texts)
tokenizer.save("tokenizer.model")
print(f"Tokenizador atualizado: {tokenizer.get_vocab_size()} tokens")

# =============================
# FINE-TUNING RÁPIDO
# =============================
print("Iniciando fine-tuning com novos dados...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for item in new_interactions:
    prompt = f"Q: {item['question']}\nA:"
    target = item["answer"]
    
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    label_ids = torch.tensor([tokenizer.encode(target)], dtype=torch.long)
    
    # Padding
    input_ids = torch.nn.utils.rnn.pad_sequence([input_ids[0]], batch_first=True, padding_value=0)
    label_ids = torch.nn.utils.rnn.pad_sequence([label_ids[0]], batch_first=True, padding_value=-100)
    
    # Forward
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), label_ids.view(-1), ignore_index=-100)
    
    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"  → Aprendido: '{item['question']}' → '{item['answer']}' | Loss: {loss.item():.4f}")

# =============================
# SALVAR TUDO
# =============================
print("Salvando modelo atualizado...")
torch.save(model.state_dict(), "snaxia_model.pth")
tokenizer.save("tokenizer.model")

# Log
log_entry = {
    "timestamp": datetime.now().isoformat(),
    "new_examples": len(new_interactions),
    "vocab_size": tokenizer.get_vocab_size(),
    "status": "success"
}

with open("training_log.json", "a") as f:
    f.write(json.dumps(log_entry) + "\n")

print("SnaX IA atualizada com sucesso!")
print(f"   Novo vocabulário: {tokenizer.get_vocab_size()} tokens")
print(f"   Log salvo em: training_log.json")