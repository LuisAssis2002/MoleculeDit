import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import MolStandardize
from smiles_representation import SmilesTokenizer
from tqdm import tqdm
import os

def clean_and_filter_chembl_data(
    raw_csv_path="data/raw/chembl_egfr_raw.csv",
    output_txt_path="data/processed/egfr_actives_cleaned.txt",
    activity_threshold=1000.0
):
    """
    PASSO 1: Lê o ficheiro CSV bruto do ChEMBL, filtra por compostos ativos,
    limpa e padroniza os SMILES, e salva o resultado num ficheiro de texto limpo.
    """
    print("--- Iniciando Passo 1: Limpeza e Filtragem dos Dados Brutos ---")
    
    # Verificar se o ficheiro de entrada existe
    if not os.path.exists(raw_csv_path):
        print(f"ERRO: Ficheiro de entrada não encontrado em '{raw_csv_path}'")
        print("Por favor, baixe o CSV do ChEMBL e coloque-o no diretório 'data/raw/'.")
        return

    # Carregar dados
    df = pd.read_csv(raw_csv_path, sep=';')
    print(f"Carregados {len(df)} registos de bioatividade.")

    # Filtragem básica
    df.dropna(subset=['Smiles', 'Standard Value'], inplace=True)
    df = df[df['Standard Units'] == 'nM']
    print(f"Restam {len(df)} registos após remover valores nulos e filtrar por unidades 'nM'.")

    # Converter valor para numérico
    df['Standard Value'] = pd.to_numeric(df['Standard Value'])

    # Filtrar por atividade
    ativos_df = df[df['Standard Value'] < activity_threshold].copy()
    print(f"Encontrados {len(ativos_df)} registos ativos (IC50 < {activity_threshold} nM).")

    # Remover duplicatas
    ativos_df.drop_duplicates(subset=['Smiles'], inplace=True)
    print(f"Restam {len(ativos_df)} compostos únicos.")

    # Funções de padronização com RDKit
    remover = SaltRemover()
    normalizer = MolStandardize.normalize.Normalizer()
    lfc = MolStandardize.fragment.LargestFragmentChooser()

    def padronizar_molecula(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            mol = remover.StripMol(mol)
            mol = lfc.choose(mol)
            mol = normalizer.normalize(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None

    # Aplicar padronização
    print("Padronizando SMILES com RDKit (isto pode demorar)...")
    tqdm.pandas()
    ativos_df['smiles_canonico'] = ativos_df['Smiles'].progress_apply(padronizar_molecula)
    
    # Limpeza final
    ativos_df.dropna(subset=['smiles_canonico'], inplace=True)
    ativos_df.drop_duplicates(subset=['smiles_canonico'], inplace=True)
    print(f"Número final de moléculas limpas e padronizadas: {len(ativos_df)}")
    
    # Salvar o ficheiro de texto limpo
    smiles_finais = ativos_df['smiles_canonico']
    smiles_finais.to_csv(output_txt_path, index=False, header=False)
    
    print(f"--- Passo 1 concluído. Ficheiro de SMILES limpos salvo em '{output_txt_path}' ---\n")


def create_tokenized_dataset(
    input_txt_path="data/processed/egfr_actives_cleaned.txt",
    output_pt_path="data/processed/egfr_actives_tokenized.pt",
    max_len=200
):
    """
    PASSO 2: Lê o ficheiro de texto com SMILES limpos, tokeniza-os, aplica padding e
    salva o resultado como um tensor PyTorch, pronto para o treino.
    """
    print("--- Iniciando Passo 2: Tokenização e Criação do Tensor ---")
    
    # Verificar se o ficheiro de entrada existe
    if not os.path.exists(input_txt_path):
        print(f"ERRO: Ficheiro de entrada não encontrado em '{input_txt_path}'")
        print("Por favor, execute primeiro a função 'clean_and_filter_chembl_data' para gerar este ficheiro.")
        return

    # Instanciar o tokenizer
    tokenizer = SmilesTokenizer()
    pad_idx = tokenizer.char_to_idx["<pad>"]
    print(f"Vocabulário construído com {tokenizer.vocab_size} tokens.")

    # Ler o ficheiro de SMILES
    with open(input_txt_path, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]
    print(f"Lidos {len(smiles_list)} SMILES do ficheiro de entrada.")

    # Tokenizar e aplicar padding
    all_tokens = []
    print(f"Tokenizando e aplicando padding para um comprimento máximo de {max_len}...")
    for smiles in tqdm(smiles_list):
        tokens = tokenizer.encode(smiles)
        if len(tokens) < max_len:
            tokens.extend([pad_idx] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len-1] + [tokenizer.char_to_idx["<end>"]]
        all_tokens.append(tokens)

    # Converter para tensor PyTorch
    dataset_tensor = torch.tensor(all_tokens, dtype=torch.long)
    print(f"\nForma do tensor final: {dataset_tensor.shape}")

    # Salvar o tensor
    torch.save(dataset_tensor, output_pt_path)
    print(f"--- Passo 2 concluído. Dataset tokenizado salvo com sucesso em: '{output_pt_path}' ---")


if __name__ == "__main__":
    # =============================================================================
    # --- CONTROLO DE EXECUÇÃO ---
    # Descomente a função que deseja executar.
    # Execute o PASSO 1 primeiro. Depois de ter o ficheiro de texto limpo,
    # pode comentar o PASSO 1 e executar o PASSO 2.
    # =============================================================================

    # PASSO 1: Limpar e filtrar os dados brutos do ChEMBL.
    # Requer: 'data/raw/chembl_egfr_raw.csv'
    # Produz: 'data/processed/egfr_actives_cleaned.txt'
    clean_and_filter_chembl_data()
    
    # PASSO 2: Converter o texto limpo num tensor para o treino.
    # Requer: 'data/processed/egfr_actives_cleaned.txt'
    # Produz: 'data/processed/egfr_actives_tokenized.pt'
    #create_tokenized_dataset()
