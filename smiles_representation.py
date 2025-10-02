import torch
import torch.nn as nn
import math

class SmilesTokenizer:
    """
    Classe para tokenizar e de-tokenizar strings SMILES.
    Versão 3.0: Constrói o vocabulário dinamicamente a partir dos dados.
    """
    def __init__(self):
        self.vocab = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def fit_on_smiles_list(self, smiles_list):
        """
        Constrói o vocabulário a partir de uma lista de todas as strings SMILES do dataset.
        """
        all_chars = set()
        for smiles in smiles_list:
            all_chars.update(list(smiles))
        
        special_tokens = ["<pad>", "<start>", "<end>"]
        self.vocab = special_tokens + sorted(list(all_chars))
        
        self.char_to_idx = {char: i for i, char in enumerate(self.vocab)}
        self.idx_to_char = {i: char for i, char in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
        print(f"Vocabulário construído com {self.vocab_size} tokens a partir dos dados.")

    def encode(self, smiles):
        """Converte uma string SMILES numa lista de tokens inteiros."""
        tokens = [self.char_to_idx["<start>"]]
        tokens.extend([self.char_to_idx.get(char, 0) for char in smiles])
        tokens.append(self.char_to_idx["<end>"])
        return tokens

    def decode(self, tokens):
        """Converte uma lista/array de tokens inteiros de volta para uma string SMILES."""
        chars = []
        for token_idx in tokens:
            char = self.idx_to_char.get(int(token_idx))
            if char == '<end>' or char == '<pad>':
                break
            if char != '<start>':
                chars.append(char)
        return "".join(chars)


class PositionalEncoding(nn.Module):
    """Adiciona informação posicional aos embeddings dos tokens."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SmilesEmbedding(nn.Module):
    """Módulo completo que combina o embedding de tokens e a codificação posicional."""
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.d_model = d_model

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return x

