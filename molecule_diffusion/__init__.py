"""
Este ficheiro atua como a porta de entrada para o nosso módulo de difusão.
Ele importa os componentes essenciais dos outros ficheiros e define
funções de conveniência para o resto do nosso projeto.
"""
from .gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_named_beta_schedule,
)

def create_diffusion(
    num_diffusion_timesteps=1000,
    beta_schedule="linear",
    mean_type=ModelMeanType.EPSILON,  # O nosso modelo irá prever o ruído (epsilon)
    loss_type=LossType.MSE,           # A nossa perda será o Erro Quadrático Médio
):
    """
    Função de conveniência ('helper function') para criar e configurar o objeto GaussianDiffusion.
    Isto evita ter de repetir esta configuração no nosso script de treino.
    """
    # 1. Criar o 'beta schedule' (define o quão agressivamente o ruído é adicionado)
    betas = get_named_beta_schedule(beta_schedule, num_diffusion_timesteps)

    # 2. Instanciar o objeto de difusão principal com as nossas definições
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=mean_type,
        model_var_type=ModelVarType.FIXED_SMALL,  # Configuração padrão, a variância é fixa
        loss_type=loss_type,
    )

    return diffusion

