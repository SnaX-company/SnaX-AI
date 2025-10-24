# tokenizer.py
# Tokenizador Adaptativo da SnaX IA
# - Baseado em SentencePiece (BPE)
# - Aprende novas palavras automaticamente
# - Nunca precisa de retraining completo
# - Roda em celular (offline)

import sentencepiece as spm
from collections import Counter
import os
import json

class SnaXTokenizer:
    """
    SnaX IA Tokenizer — Tokenizador que evolui com o uso
    - Detecta palavras novas
    - Atualiza vocabulário automaticamente
    - Usa byte_fallback → nunca tem <unk>
    """
    def __init__(self, model_path="tokenizer.model", vocab_size=752):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.sp = spm.SentencePieceProcessor()
        self.buffer = Counter()      # Armazena novos tokens candidatos
        self.update_threshold = 50   # Atualiza se >50 ocorrências
        self.min_freq_add = 20       # Só adiciona se aparecer 20x
        
        # Se não existe, cria tokenizador inicial
        if not os.path.exists(model_path):
            self._create_initial_tokenizer()
        else:
            self.sp.load(model_path)
            print(f"SnaX Tokenizer carregado: {self.sp.get_piece_size()} tokens")

    def _create_initial_tokenizer(self):
        """Cria tokenizador inicial com texto básico em português"""
        init_corpus = """
        SnaX IA é uma inteligência artificial portátil. 
        Ela roda em celular, aprende sozinha e raciocina como humano.
        Pode resolver problemas de matemática, lógica e linguagem.
        Usa redes neurais espiculantes (SNNs) para eficiência energética.
        123 números, emojis, código Python, tudo funciona perfeitamente.
        O futuro da IA está no bolso.
        """
        with open("init_corpus.txt", "w", encoding="utf-8") as f:
            f.write(init_corpus)
        
        spm.SentencePieceTrainer.train(
            input="init_corpus.txt",
            model_prefix="tokenizer",
            vocab_size=self.vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            byte_fallback=True,           # Crucial: nunca gera <unk>
            split_digits=True,
            add_dummy_prefix=True,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3
        )
        self.sp.load("tokenizer.model")
        print(f"SnaX Tokenizer inicial criado: {self.sp.get_piece_size()} tokens")

    # ======================
    # MÉTODOS BÁSICOS
    # ======================
    def encode(self, text):
        """Converte texto → lista de IDs"""
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        """Converte IDs → texto"""
        return self.sp.decode(ids)

    def get_vocab_size(self):
        return self.sp.get_piece_size()

    # ======================
    # ATUALIZAÇÃO AUTOMÁTICA
    # ======================
    def update_with_texts(self, texts):
        """
        Chame isso com uma lista de textos novos (ex: perguntas do usuário)
        Ele detecta palavras desconhecidas e atualiza o vocabulário
        """
        unk_tokens = []
        raw_words = []
        
        for text in texts:
            encoded = self.sp.encode(text, out_type=str)
            # Detecta <unk>
            unks = [t for t in encoded if t == "<unk>"]
            if unks:
                # Extrai palavras reais do texto
                words = text.lower().split()
                raw_words.extend(words)
                unk_tokens.extend(unks)
        
        # Atualiza buffer
        self.buffer.update(raw_words)
        
        # Se houver muitos <unk>, atualiza
        if len(unk_tokens) > self.update_threshold:
            self._trigger_vocab_update()

    def _trigger_vocab_update(self):
        """Atualiza o vocabulário com tokens frequentes"""
        # Filtra apenas os mais frequentes
        candidates = {k: v for k, v in self.buffer.items() if v >= self.min_freq_add}
        if not candidates:
            return
        
        print(f"SnaX Tokenizer: Atualizando com {len(candidates)} novos tokens...")
        
        # Salva candidatos em arquivo temporário
        with open("update_corpus.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(candidates.keys()))
        
        # Novo tamanho de vocabulário
        new_vocab_size = min(self.sp.get_piece_size() + len(candidates), 32000)
        
        # Retreina com merge
        spm.SentencePieceTrainer.train(
            input=["init_corpus.txt", "update_corpus.txt"],
            model_prefix="tokenizer_new",
            vocab_size=new_vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            byte_fallback=True,
            split_digits=True
        )
        
        # Carrega novo modelo
        self.sp.load("tokenizer_new.model")
        self.buffer.clear()
        
        print(f"SnaX Tokenizer atualizado! Novo tamanho: {self.sp.get_piece_size()} tokens")

    # ======================
    # UTILIDADES
    # ======================
    def save(self, path="tokenizer.model"):
        """Salva modelo para usar no celular"""
        if os.path.exists("tokenizer_new.model"):
            os.replace("tokenizer_new.model", path)
        elif os.path.exists("tokenizer.model"):
            os.replace("tokenizer.model", path)

    def __len__(self):
        return self.sp.get_piece_size()