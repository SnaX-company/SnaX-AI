# train_v3.py
# Treinamento REALISTA da SnaX IA v3
# - Dataset real (Wikipedia PT + código)
# - Treinamento eficiente com mixed precision
# - Checkpoints e logging
# - Roda em 2-3h no Colab grátis

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from model_v3 import SnaXIA_v3
from tokenizer_v3 import SnaXTokenizer_v3
import os
import json
from tqdm import tqdm
import wandb  # pip install wandb (opcional, para tracking)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
CONFIG = {
    # Modelo
    "vocab_size": 8192,
    "hidden_size": 384,
    "num_layers": 6,
    "num_heads": 6,
    "num_kv_heads": 2,
    "max_seq_len": 512,
    
    # Treinamento
    "batch_size": 16,
    "grad_accum_steps": 4,  # Batch efetivo = 16*4 = 64
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 500,
    "max_steps": 10000,
    "eval_interval": 500,
    "save_interval": 1000,
    
    # Sistema
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mixed_precision": True,
    "compile_model": False,  # PyTorch 2.0+ compile (mais rápido)
    
    # Paths
    "checkpoint_dir": "checkpoints",
    "use_wandb": False,  # Ative para tracking online
}

print("SnaX IA v3 - Treinamento")
print("=" * 50)
print(f"Device: {CONFIG['device']}")
print(f"Mixed Precision: {CONFIG['mixed_precision']}")

# =============================================================================
# DATASET
# =============================================================================
class TextDataset(Dataset):
    """Dataset simples para treinamento de linguagem"""
    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokeniza
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        
        # Trunca
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Para LM, input = target deslocado
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch):
    """Padding dinâmico"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Pad
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=3)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {"input_ids": input_ids, "labels": labels}

def load_training_data():
    """
    Carrega dataset de treinamento
    Opções:
    1. Wikipedia PT (pequeno, mas bom)
    2. Carolina (código em português)
    3. Seu próprio dataset
    """
    print("Carregando dataset...")
    
    # Opção 1: Wikipedia PT (gratuito no HuggingFace)
    try:
        dataset = load_dataset("graelo/wikipedia", "20230601.pt", split="train[:10000]")
        texts = [item["text"] for item in dataset if len(item["text"]) > 100]
        print(f"✅ Carregados {len(texts)} textos da Wikipedia PT")
        return texts
    except:
        print("⚠️  Erro ao baixar dataset. Usando fallback...")
    
    # Opção 2: Fallback - corpus sintético
    fallback_texts = [
        "Python é uma linguagem de programação de alto nível.",
        "Machine learning utiliza algoritmos para aprender padrões.",
        "A inteligência artificial está revolucionando diversas áreas.",
        "Redes neurais são inspiradas no funcionamento do cérebro.",
    ] * 1000  # Repete para simular dataset
    
    print(f"⚠️  Usando {len(fallback_texts)} textos sintéticos")
    return fallback_texts

# =============================================================================
# TREINAMENTO
# =============================================================================
def train():
    # Cria diretório de checkpoints
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    
    # Inicializa W&B (opcional)
    if CONFIG["use_wandb"]:
        wandb.init(project="snax-ia-v3", config=CONFIG)
    
    # Tokenizador
    print("\nInicializando tokenizador...")
    tokenizer = SnaXTokenizer_v3()
    CONFIG["vocab_size"] = tokenizer.vocab_size
    
    # Modelo
    print("\nInicializando modelo...")
    model = SnaXIA_v3(CONFIG)
    model = model.to(CONFIG["device"])
    
    print(f"Parâmetros: {model.get_num_params():,}")
    
    # Compile (PyTorch 2.0+)
    if CONFIG["compile_model"] and hasattr(torch, "compile"):
        print("Compilando modelo...")
        model = torch.compile(model)
    
    # Dataset
    texts = load_training_data()
    train_size = int(0.95 * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    
    train_dataset = TextDataset(tokenizer, train_texts, CONFIG["max_seq_len"])
    val_dataset = TextDataset(tokenizer, val_texts, CONFIG["max_seq_len"])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        collate_fn=collate_fn
    )
    
    # Otimizador
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.95)
    )
    
    # Scheduler com warmup
    def get_lr(step):
        if step < CONFIG["warmup_steps"]:
            return step / CONFIG["warmup_steps"]
        return max(0.1, 1.0 - (step - CONFIG["warmup_steps"]) / (CONFIG["max_steps"] - CONFIG["warmup_steps"]))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    
    # Mixed precision
    scaler = GradScaler() if CONFIG["mixed_precision"] else None
    
    # Loop de treinamento
    print("\nIniciando treinamento...")
    print(f"Batch size efetivo: {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")
    
    model.train()
    global_step = 0
    running_loss = 0
    
    for epoch in range(100):  # Vai parar por max_steps
        pbar = tqdm(train_loader, desc=f"Época {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(CONFIG["device"])
            labels = batch["labels"].to(CONFIG["device"])
            
            # Forward
            with autocast(enabled=CONFIG["mixed_precision"]):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, CONFIG["vocab_size"]),
                    labels.view(-1),
                    ignore_index=-100
                )
                loss = loss / CONFIG["grad_accum_steps"]
            
            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            running_loss += loss.item()
            
            # Update
            if (batch_idx + 1) % CONFIG["grad_accum_steps"] == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                avg_loss = running_loss * CONFIG["grad_accum_steps"]
                running_loss = 0
                
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
                
                # Log
                if CONFIG["use_wandb"]:
                    wandb.log({"train_loss": avg_loss, "lr": scheduler.get_last_lr()[0]}, step=global_step)
                
                # Validação
                if global_step % CONFIG["eval_interval"] == 0:
                    val_loss = evaluate(model, val_loader, CONFIG)
                    print(f"\n[Step {global_step}] Val Loss: {val_loss:.4f}")
                    
                    if CONFIG["use_wandb"]:
                        wandb.log({"val_loss": val_loss}, step=global_step)
                    
                    model.train()
                
                # Checkpoint
                if global_step % CONFIG["save_interval"] == 0:
                    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"checkpoint_{global_step}.pt")
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": CONFIG,
                    }, checkpoint_path)
                    print(f"✅ Checkpoint salvo: {checkpoint_path}")
                
                # Para se atingir max_steps
                if global_step >= CONFIG["max_steps"]:
                    print(f"\n✅ Treinamento concluído! ({CONFIG['max_steps']} steps)")
                    
                    # Salva modelo final
                    final_path = "snaxia_v3_final.pt"
                    torch.save(model.state_dict(), final_path)
                    tokenizer.save("tokenizer_v3_final.model")
                    
                    print(f"Modelo salvo: {final_path}")
                    return
        
        if global_step >= CONFIG["max_steps"]:
            break

@torch.no_grad()
def evaluate(model, val_loader, config):
    """Avalia modelo no conjunto de validação"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in val_loader:
        input_ids = batch["input_ids"].to(config["device"])
        labels = batch["labels"].to(config["device"])
        
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, config["vocab_size"]),
            labels.view(-1),
            ignore_index=-100
        )
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches >= 20:  # Valida só 20 batches (rápido)
            break
    
    return total_loss / num_batches

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️  Treinamento interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
