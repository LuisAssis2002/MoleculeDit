"""
Este ficheiro de inicialização torna mais fácil importar as classes e funções
essenciais do módulo de difusão para outras partes do nosso projeto.

Este código foi adaptado do repositório CrystalDiT, que por sua vez o adaptou
de repositórios da OpenAI e da Meta. A lógica matemática do processo de difusão
é genérica e pode ser aplicada diretamente ao nosso problema de geração de moléculas.
"""

# Importa a função principal para criar o "agendamento de ruído" (noise schedule).
# Esta função define a quantidade de ruído a ser adicionada em cada passo de tempo.
# 'linear' e 'squaredcos_cap_v2' são os mais comuns.
from .gaussian_diffusion import get_named_beta_schedule

# A classe principal que gere todo o processo de difusão gaussiana.
# Ela contém a matemática para o processo forward (adicionar ruído) e
# as funções para o processo reverse (gerar novas amostras).
from .gaussian_diffusion import GaussianDiffusion

# Importa os enums que definem o comportamento do modelo de difusão, como
# o que o modelo deve prever (o ruído 'EPSILON' é o padrão moderno) e como
# a perda (loss) é calculada.
from .gaussian_diffusion import ModelMeanType, ModelVarType, LossType

# Esta classe permite acelerar a geração (amostragem) ao "saltar" passos
# de tempo, em vez de passar por todos os 1000. É muito útil para gerar
# amostras rapidamente.
from .respace import SpacedDiffusion, space_timesteps

# O 'UniformSampler' é a forma mais simples de escolher um passo de tempo 't'
# durante o treino: cada passo de tempo tem a mesma probabilidade de ser escolhido.
from .timestep_sampler import UniformSampler
