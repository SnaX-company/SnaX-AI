# SnaX IA — Inteligência Artificial com Raciocínio no Seu Bolso

> **Spiking Neural Architecture eXtreme**  
> A primeira IA que **roda em qualquer celular**, **aprende sozinha** e **raciocina como humano** — **100% grátis e open-source**.

---

## Recursos

- **Roda em Android/iOS** (15MB)
- **Raciocínio matemático, lógico e linguístico**
- **Tokenizador auto-atualizável** (aprende gírias, números, emojis)
- **Treinamento contínuo** (GitHub Actions grátis)
- **Escalável** de 500M → 10B+ parâmetros
- **Zero custo** — tudo com Colab, Kaggle, GitHub

---

## Arquivos do Projeto

| Arquivo | Função |
|-------|--------|
| `model.py` | Arquitetura SnaX IA (SNN + Transformer) |
| `tokenizer.py` | Tokenizador que evolui sozinho |
| `main.py` | Treinamento inicial (GSM8K) |
| `optimize_mobile.py` | Exporta para celular (15MB) |
| `train_incremental.py` | Aprendizado contínuo diário |
| `evaluate.py` | Chat interativo com a IA |
| `README.md` | Este arquivo |

---

## Como Rodar (1 Clique no Colab)

1. Abra: [https://colab.research.google.com](https://colab.research.google.com)
2. Cole isso e rode:

```python
# 1. Instalar dependências
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers datasets sentencepiece snntorch

# 2. Baixar projeto
!git clone https://github.com/SEU_USUARIO/snax_ia
%cd snax_ia

# 3. Treinar
!python main.py

# 4. Otimizar para celular
!python optimize_mobile.py

# 5. Testar
!python evaluate.py