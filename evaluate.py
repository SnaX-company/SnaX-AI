# evaluate.py
# Testador interativo da SnaX IA
# - Carrega modelo + tokenizador
# - Responde perguntas em tempo real
# - Roda no PC ou Colab

import torch
from model import SnaXIA
from tokenizer import SnaXTokenizer
import os

print("SnaX IA: Iniciando testador interativo...")
print("=" * 50)

# =============================
# CARREGAR MODELO E TOKENIZADOR
# =============================
if not os.path.exists("snaxia_model.pth"):
    print("ERRO: snaxia_model.pth não encontrado!")
    print("Rode main.py primeiro para treinar o modelo.")
    exit()

if not os.path.exists("tokenizer.model"):
    print("ERRO: tokenizer.model não encontrado!")
    exit()

print("Carregando SnaX Tokenizer...")
tokenizer = SnaXTokenizer()
VOCAB_SIZE = tokenizer.get_vocab_size()

print("Carregando SnaX IA...")
model = SnaXIA(vocab_size=VOCAB_SIZE)
model.load_state_dict(torch.load("snaxia_model.pth", map_location="cpu"))
model.eval()

print(f"SnaX IA pronta! Vocabulário: {VOCAB_SIZE} tokens")
print("=" * 50)

# =============================
# FUNÇÃO DE PERGUNTA
# =============================
def ask_snax(question, max_tokens=50):
    """
    Faz uma pergunta à SnaX IA
    """
    prompt = f"Q: {question}\nA:"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    
    print(f"\nPergunta: {question}")
    print("SnaX IA pensando", end="")
    
    generated = input_ids[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            
            # Para no fim
            if next_token == tokenizer.sp.piece_to_id("<|eot|>"):
                break
            
            print(".", end="", flush=True)
    
    response = tokenizer.decode(generated)
    answer = response.split("A:")[-1].strip()
    
    print(f"\nResposta: {answer}")
    print("-" * 50)
    return answer

# =============================
# TESTES PRÉ-DEFINIDOS
# =============================
print("TESTES AUTOMÁTICOS:")
test_questions = [
    "Quanto é 47 + 85?",
    "Se um trem sai às 14h e chega às 17h30, quanto tempo durou?",
    "Qual é a capital do Brasil?",
    "O que é inteligência artificial?",
    "Explique em 1 frase: por que a SnaX IA é diferente?"
]

for q in test_questions:
    ask_snax(q)

# =============================
# MODO INTERATIVO
# =============================
print("\nMODO INTERATIVO (digite 'sair' para encerrar)")
while True:
    user_input = input("\nSua pergunta: ").strip()
    if user_input.lower() in ["sair", "exit", "quit"]:
        print("SnaX IA: Até logo!")
        break
    if user_input:
        ask_snax(user_input)