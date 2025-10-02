# Este ficheiro define a arquitetura principal do nosso Diffusion Transformer para Moléculas.
# Ele foi adaptado a partir do ficheiro `models.py` do repositório DiT/CrystalDiT.
#
# Principais Adaptações:
# 1. REMOVIDO: `PatchEmbed` e `LabelEmbedder`, que são específicos para imagens e classes.
# 2. ADICIONADO: `SmilesEmbedding`, o nosso módulo personalizado para processar SMILES.
# 3. SIMPLIFICADO: A classe `DiT` foi renomeada para `MoleculeDiT` e simplificada para
#    aceitar apenas sequências de tokens (x) e timesteps (t), sem rótulos de classe (y).
# 4. ADAPTADO: A `FinalLayer` foi simplificada para produzir uma saída com a mesma forma
#    da sequência de embeddings de entrada.

import torch
import torch.nn as nn
import math

# --- Peças Reutilizadas do models.py ---
# Estas classes e funções são genéricas e podem ser usadas diretamente.
# Elas formam os blocos de construção internos do nosso Transformer.

from timm.models.vision_transformer import Mlp, Attention
from smiles_representation import SmilesEmbedding # A nossa camada de input!

def modulate(x, shift, scale):
    """Função auxiliar para aplicar a modulação AdaLN (Adaptive Layer Norm)."""
    # A modulação acontece após a normalização de camada (LayerNorm).
    # Ela permite que a informação do timestep 't' condicione a ativação.
    # Shape de x: (batch, seq_len, hidden_size)
    # Shape de shift/scale: (batch, hidden_size)
    # Precisamos de expandir shift/scale para corresponder à forma de x.
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Converte timesteps escalares (ex: 257) em representações vetoriais ricas.
    Usa embeddings senoidais de frequência, seguidos por um MLP.
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
        """Cria os embeddings senoidais."""
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
    Um bloco do Diffusion Transformer com condicionamento adaLN-Zero.
    Este é o "músculo" do modelo, contendo auto-atenção e um MLP.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # A modulação gera 6 vetores a partir do embedding do timestep 'c'.
        # Estes vetores (shift, scale, gate) são usados para condicionar as camadas de atenção e MLP.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Caminho da Auto-Atenção
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # Caminho do MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    A camada final do nosso MoleculeDiT.
    Aplica uma modulação final e uma projeção linear para obter a previsão do ruído.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True) # A saída tem a mesma dimensão que o embedding
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# --- A Nossa Arquitetura Principal ---

class MoleculeDiT(nn.Module):
    """
    Diffusion model com um backbone Transformer para geração de moléculas.
    """
    def __init__(self, vocab_size, hidden_size=512, max_len=200, depth=12, num_heads=8):
        super().__init__()
        
        # 1. Camada de Embedding para SMILES (A nossa grande adaptação!)
        self.embedding = SmilesEmbedding(vocab_size, hidden_size, max_len)

        # 2. Embedding para o passo de tempo (reutilizado)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # 3. Corpo do Transformer com 'depth' blocos DiT (reutilizado)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # 4. Camada final para prever o ruído (a nossa versão simplificada)
        self.final_layer = FinalLayer(hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        """Inicializa os pesos de forma a otimizar a estabilidade do treino."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Inicializar o embedding do timestep
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zerar as camadas de modulação adaLN nos blocos DiT
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zerar a camada de saída final
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t):
        """
        Forward pass do MoleculeDiT.
        :param x: Tensor de tokens de SMILES ruidosos. Shape: (batch, seq_len)
        :param t: Tensor de timesteps. Shape: (batch,)
        """
        # 1. Converter tokens (x) e timestep (t) em embeddings
        # A nossa camada de embedding lida com tokens + posição
        x_embed = self.embedding(x)  # Shape: (batch, seq_len, hidden_size)
        t_embed = self.t_embedder(t) # Shape: (batch, hidden_size)

        # 2. Passar pelos blocos do Transformer
        for block in self.blocks:
            x_embed = block(x_embed, t_embed)

        # 3. Prever o ruído a partir da saída final
        noise_prediction = self.final_layer(x_embed, t_embed) # Shape: (batch, seq_len, hidden_size)
        
        return noise_prediction

