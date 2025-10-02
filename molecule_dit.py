import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, Attention

# Importar o nosso módulo de representação de SMILES
from smiles_representation import SmilesEmbedding


def modulate(x, shift, scale):
    """
    Função auxiliar para aplicar a modulação AdaLN (Adaptive Layer Norm).
    Esta função ajusta as ativações da rede com base no timestep.
    """
    # A modulação expande shift e scale para terem a mesma dimensão de x
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Converte um timestep escalar (ex: 257) num vetor de embedding de alta dimensão.
    Este código é reutilizado do projeto DiT original.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Cria embeddings sinusoidais para os timesteps."""
        import math
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    Um bloco do Transformer (DiT Block) com condicionamento AdaLN-Zero.
    Este é o "cérebro" da rede, onde a auto-atenção acontece.
    Este código é reutilizado do projeto DiT original.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    A camada final do nosso MoleculeDiT.
    A sua função é pegar nos embeddings processados e prepará-los para a saída.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True) # A saída são embeddings
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MoleculeDiT(nn.Module):
    """
    A nossa arquitetura principal, adaptada do DiT para gerar moléculas a partir de SMILES.
    """
    def __init__(
        self,
        vocab_size,
        max_len=200,
        hidden_size=512,
        depth=12,
        num_heads=8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len

        # --- ALTERAÇÃO IMPORTANTE ---
        # A camada de embedding é definida aqui, mas NÃO será chamada no `forward` principal.
        # Será chamada explicitamente no loop de treino.
        self.embedding = SmilesEmbedding(vocab_size, hidden_size, max_len)
        # ---------------------------

        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size)
        
        # Camada de projeção para a avaliação
        self.projection_head = nn.Linear(hidden_size, vocab_size)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Inicializa os pesos de forma semelhante ao DiT original
        self.apply(self._init_weights)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, t):
        """
        Forward pass do MoleculeDiT.
        
        Args:
            x (torch.Tensor): Tensor de embeddings de entrada (Float). 
                              Forma: [batch_size, seq_len, hidden_size]
            t (torch.Tensor): Tensor de timesteps. Forma: [batch_size]
        """
        # --- ALTERAÇÃO IMPORTANTE ---
        # `x` já é o embedding. Não chamamos a camada de embedding aqui.
        # ---------------------------
        
        t_emb = self.t_embedder(t)  # Embedding do timestep
        
        # O timestep é o único condicionamento
        c = t_emb

        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_layer(x, c)
        return x # A saída é uma previsão de ruído no espaço dos embeddings

    def project_to_tokens(self, x):
        """
        Pega nos embeddings finais e projeta-os para o espaço do vocabulário.
        Usado apenas na avaliação/geração.
        """
        return self.projection_head(x)

