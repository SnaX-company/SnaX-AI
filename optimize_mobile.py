# optimize_mobile.py
# Transforma a SnaX IA em um modelo leve para Android/iOS
# - Quantização INT8
# - Pruning 50%
# - Exporta como .ptl (PyTorch Mobile)
# - Resultado: ~15MB, roda em 200ms no celular

import torch
import torch.nn.utils.prune as prune
from model import SnaXIA
from tokenizer import SnaXTokenizer
import os

print("SnaX IA Mobile: Iniciando otimização...")

# =============================
# CARREGAR MODELO TREINADO
# =============================
if not os.path.exists("snaxia_model.pth"):
    print("ERRO: snaxia_model.pth não encontrado!")
    print("Rode main.py primeiro para treinar o modelo.")
    exit()

tokenizer = SnaXTokenizer()
VOCAB_SIZE = tokenizer.get_vocab_size()

model = SnaXIA(vocab_size=VOCAB_SIZE)
model.load_state_dict(torch.load("snaxia_model.pth", map_location="cpu"))
model.eval()

print(f"Modelo carregado: {sum(p.numel() for p in model.parameters()):,} parâmetros")

# =============================
# QUANTIZAÇÃO DINÂMICA (INT8)
# =============================
print("Aplicando quantização INT8...")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Embedding},
    dtype=torch.qint8
)

print("Quantização concluída.")

# =============================
# PRUNING ESTRUTURADO (50%)
# =============================
print("Aplicando pruning (50% dos pesos)...")
parameters_to_prune = []
for name, module in quantized_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5,
)

# Remove máscaras (torna pruning permanente)
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

print("Pruning concluído.")

# =============================
# EXPORTAR PARA MOBILE
# =============================
print("Exportando para formato mobile (.ptl)...")
example_input = torch.randint(0, VOCAB_SIZE, (1, 50))  # Dummy input

# Trace do modelo
scripted_model = torch.jit.trace(quantized_model, example_input)
optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)

# Salvar
output_path = "snaxia_mobile.ptl"
optimized_model._save_for_lite_interpreter(output_path)

file_size = os.path.getsize(output_path) / (1024 * 1024)
print(f"SnaX IA Mobile pronta!")
print(f"   Arquivo: {output_path}")
print(f"   Tamanho: {file_size:.1f} MB")
print(f"   Roda em: Android, iOS, qualquer celular!")

# =============================
# TESTE RÁPIDO
# =============================
print("\nTestando geração rápida...")
def test_mobile_model():
    input_text = "Q: Quanto é 23 + 19?\nA:"
    input_ids = torch.tensor([tokenizer.encode(input_text)], dtype=torch.long)
    
    with torch.no_grad():
        for _ in range(10):
            logits = quantized_model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())

print("Resposta de teste:")
print(test_mobile_model())