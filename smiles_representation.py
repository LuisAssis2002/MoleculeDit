import torch
import torch.nn as nn
import math
import numpy as np

# --- FASE 1: O Tokenizer ---
class SmilesTokenizer:
    """
    Classe para tokenizar e de-tokenizar strings SMILES.
    Funciona como o nosso "dicionário" químico.
    """
    def __init__(self, vocab=None):
        self.vocab = vocab if vocab is not None else self._build_default_vocab()
        self.char_to_idx = {char: i for i, char in enumerate(self.vocab)}
        self.idx_to_char = {i: char for i, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def _build_default_vocab(self):
        chars = " C#()[]-+=cnosFNClBrI/\\@."
        special_tokens = ["<pad>", "<start>", "<end>"]
        return special_tokens + sorted(list(set(chars)))

    def encode(self, smiles_string):
        tokens = [self.char_to_idx["<start>"]]
        for char in smiles_string:
            tokens.append(self.char_to_idx.get(char, self.char_to_idx["."]))
        tokens.append(self.char_to_idx["<end>"])
        return tokens

    def decode(self, tokens):
        chars = []
        for token in tokens:
            char = self.idx_to_char.get(token)
            if char == "<end>": break
            if char not in ["<start>", "<pad>"]: chars.append(char)
        return "".join(chars)

# --- FASE 2 & 3: Módulo de Embedding (Token + Posição) ---
class PositionalEncoding(nn.Module):
    """Módulo que adiciona a codificação posicional."""
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SmilesEmbedding(nn.Module):
    """Módulo completo que combina embedding de tokens e codificação posicional."""
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        token_embed = self.token_embedding(src) * math.sqrt(self.embed_dim)
        final_embed = self.positional_encoding(token_embed)
        return final_embed
