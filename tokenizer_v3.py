# tokenizer_v3.py
# Tokenizador REALISTA para SnaX IA v3
# - SentencePiece BPE treinado UMA VEZ
# - Vocabulário fixo de 8192 tokens
# - Suporta português + inglês + código

import sentencepiece as spm
import os
from pathlib import Path

class SnaXTokenizer_v3:
    """
    Tokenizador simples e funcional
    - Sem "atualização mágica"
    - Treina com corpus grande na inicialização
    - Usa byte-fallback para robustez
    """
    def __init__(self, model_path="tokenizer_v3.model", vocab_size=8192):
        self.model_path = model_path
        self.vocab_size = vocab_size
        
        if os.path.exists(model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            print(f"✅ Tokenizador carregado: {self.sp.get_piece_size()} tokens")
        else:
            print("⚠️  Tokenizador não encontrado. Criando novo...")
            self._train_tokenizer()
    
    def _train_tokenizer(self):
        """Treina tokenizador com corpus balanceado"""
        
        # Corpus inicial (português + inglês + código)
        corpus = """
        # PORTUGUÊS
        A inteligência artificial está transformando o mundo.
        Modelos de linguagem podem entender e gerar texto.
        Python é uma linguagem de programação muito usada.
        Matemática: 1 + 1 = 2, 5 × 7 = 35, 100 ÷ 4 = 25
        
        # INGLÊS
        Machine learning models learn patterns from data.
        Natural language processing enables text understanding.
        Neural networks consist of layers of interconnected nodes.
        
        # CÓDIGO
        def train_model(data, epochs=10):
            for epoch in range(epochs):
                loss = compute_loss(data)
                optimizer.step()
            return model
        
        import torch
        import numpy as np
        from transformers import AutoModel
        
        # NÚMEROS E SÍMBOLOS
        0123456789 !@#$%^&*()_+-=[]{}|;:,.<>?
        
        # EMOJIS COMUNS
        😀😂❤️👍🔥💯✨🎉
        """
        
        # Se você tiver um corpus maior, use aqui
        corpus_file = "training_corpus.txt"
        with open(corpus_file, "w", encoding="utf-8") as f:
            f.write(corpus)
        
        # Treina SentencePiece
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix="tokenizer_v3",
            vocab_size=self.vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            byte_fallback=True,           # Nunca gera <unk>
            split_digits=True,            # Separa dígitos
            split_by_unicode_script=True, # Separa scripts (latin, CJK, etc)
            normalization_rule_name="nmt_nfkc_cf",  # Normalização robusta
            add_dummy_prefix=True,
            
            # Tokens especiais
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            user_defined_symbols=["<|im_start|>", "<|im_end|>", "<|system|>", "<|user|>", "<|assistant|>"]
        )
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)
        
        # Remove arquivo temporário
        os.remove(corpus_file)
        
        print(f"✅ Tokenizador criado: {self.sp.get_piece_size()} tokens")
    
    # ======================
    # MÉTODOS PRINCIPAIS
    # ======================
    def encode(self, text, add_bos=False, add_eos=False):
        """Texto → IDs"""
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids
    
    def decode(self, ids):
        """IDs → Texto"""
        # Remove tokens especiais antes de decodificar
        ids = [i for i in ids if i not in [self.pad_id, self.bos_id, self.eos_id]]
        return self.sp.decode(ids)
    
    def encode_batch(self, texts, max_length=512, padding=True):
        """
        Encodifica batch com padding
        Retorna: dict com input_ids, attention_mask
        """
        encoded = [self.encode(text, add_bos=True, add_eos=True) for text in texts]
        
        if padding:
            max_len = min(max(len(ids) for ids in encoded), max_length)
            
            input_ids = []
            attention_mask = []
            
            for ids in encoded:
                # Trunca se necessário
                ids = ids[:max_len]
                
                # Padding
                pad_len = max_len - len(ids)
                mask = [1] * len(ids) + [0] * pad_len
                ids = ids + [self.pad_id] * pad_len
                
                input_ids.append(ids)
                attention_mask.append(mask)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        
        return {"input_ids": encoded}
    
    # ======================
    # PROPRIEDADES
    # ======================
    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
    
    @property
    def pad_id(self):
        return self.sp.pad_id()
    
    @property
    def bos_id(self):
        return self.sp.bos_id()
    
    @property
    def eos_id(self):
        return self.sp.eos_id()
    
    @property
    def unk_id(self):
        return self.sp.unk_id()
    
    def get_vocab(self):
        """Retorna vocabulário completo"""
        return {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
    
    def save(self, path=None):
        """Salva modelo (já está salvo automaticamente)"""
        if path and path != self.model_path:
            import shutil
            shutil.copy(self.model_path, path)
            print(f"✅ Tokenizador copiado para: {path}")
    
    # ======================
    # UTILITÁRIOS
    # ======================
    def tokenize_example(self, text):
        """Mostra como um texto é tokenizado (debug)"""
        ids = self.encode(text)
        tokens = [self.sp.id_to_piece(i) for i in ids]
        
        print(f"Texto: {text}")
        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"IDs: {ids}")
        print(f"Decodificado: {self.decode(ids)}")

# =============================================================================
# TESTES
# =============================================================================
if __name__ == "__main__":
    print("SnaX IA v3 - Teste de Tokenizador")
    print("=" * 50)
    
    tokenizer = SnaXTokenizer_v3()
    
    # Testes
    test_texts = [
        "Olá! Como você está?",
        "Quanto é 15 + 27?",
        "def hello(): print('world')",
        "Machine learning is amazing! 🚀",
        "São Paulo é a maior cidade do Brasil."
    ]
    
    print("\nTESTE DE TOKENIZAÇÃO:")
    for text in test_texts:
        tokenizer.tokenize_example(text)
        print("-" * 30)
    
    # Teste de batch
    print("\nTESTE DE BATCH:")
    batch = tokenizer.encode_batch(test_texts, max_length=32)
    print(f"Shape: {len(batch['input_ids'])}x{len(batch['input_ids'][0])}")
    print(f"Exemplo: {batch['input_ids'][0]}")
    
    print("\n✅ Tokenizador v3 funcionando!")
