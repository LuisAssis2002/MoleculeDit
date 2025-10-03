import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from smiles_representation import SmilesTokenizer
from tqdm import tqdm
import os
import pickle

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
    
    if not os.path.exists(raw_csv_path):
        print(f"ERRO: Ficheiro de entrada não encontrado em '{raw_csv_path}'")
        print("Por favor, baixe o CSV do ChEMBL e coloque-o no diretório 'data/raw/'.")
        return False

    df = pd.read_csv(raw_csv_path, sep=';')
    print(f"Carregados {len(df)} registos de bioatividade.")

    df.dropna(subset=['Smiles', 'Standard Value'], inplace=True)
    df = df[df['Standard Units'] == 'nM']
    print(f"Restam {len(df)} registos após remover valores nulos e filtrar por unidades 'nM'.")

    df['Standard Value'] = pd.to_numeric(df['Standard Value'], errors='coerce')
    df.dropna(subset=['Standard Value'], inplace=True)

    ativos_df = df[df['Standard Value'] < activity_threshold].copy()
    print(f"Encontrados {len(ativos_df)} registos ativos (IC50 < {activity_threshold} nM).")

    ativos_df.drop_duplicates(subset=['Smiles'], inplace=True)
    print(f"Restam {len(ativos_df)} compostos únicos.")

    remover = SaltRemover()
    normalizer = rdMolStandardize.Normalizer()
    lfc = rdMolStandardize.LargestFragmentChooser()

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

    print("Padronizando SMILES com RDKit (isto pode demorar)...")
    tqdm.pandas()
    ativos_df['smiles_canonico'] = ativos_df['Smiles'].progress_apply(padronizar_molecula)
    
    ativos_df.dropna(subset=['smiles_canonico'], inplace=True)
    ativos_df.drop_duplicates(subset=['smiles_canonico'], inplace=True)
    print(f"Número final de moléculas limpas e padronizadas: {len(ativos_df)}")
    
    smiles_finais = ativos_df['smiles_canonico']
    smiles_finais.to_csv(output_txt_path, index=False, header=False)
    
    print(f"--- Passo 1 concluído. Ficheiro de SMILES limpos salvo em '{output_txt_path}' ---\n")
    return True


def create_tokenized_dataset(
    input_txt_path="data/processed/egfr_actives_cleaned.txt",
    output_tensor_path="data/processed/egfr_actives_tokenized.pt",
    output_tokenizer_path="data/processed/tokenizer.pkl",
    max_len=200
):
    """
    PASSO 2: Lê os SMILES limpos, constrói um tokenizer, tokeniza os dados,
    e salva tanto o tensor de dados QUANTO o objeto tokenizer.
    """
    print("--- Iniciando Passo 2: Tokenização e Criação do Dataset ---")
    if not os.path.exists(input_txt_path):
        print(f"ERRO: Ficheiro de entrada não encontrado em '{input_txt_path}'")
        return False

    with open(input_txt_path, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]

    tokenizer = SmilesTokenizer()
    tokenizer.fit_on_smiles_list(smiles_list)
    
    with open(output_tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer treinado e salvo em: '{output_tokenizer_path}'")

    pad_idx = tokenizer.char_to_idx["<pad>"]
    
    all_tokens = []
    for smiles in tqdm(smiles_list, desc="Tokenizando SMILES"):
        tokens = tokenizer.encode(smiles)
        if len(tokens) < max_len:
            tokens.extend([pad_idx] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len-1] + [tokenizer.char_to_idx["<end>"]]
        all_tokens.append(tokens)

    dataset_tensor = torch.tensor(all_tokens, dtype=torch.long)
    torch.save(dataset_tensor, output_tensor_path)
    print(f"\nForma do tensor final: {dataset_tensor.shape}")
    print(f"--- Passo 2 concluído. Dataset tokenizado salvo em: '{output_tensor_path}' ---")
    return True


if __name__ == "__main__":
    # --- CONTROLO DE EXECUÇÃO ---
    # Este bloco agora executa ambas as etapas em sequência para garantir
    # que a pipeline de dados seja sempre concluída.

    # PASSO 1: Limpa os dados brutos e cria o ficheiro .txt
    success_step1 = clean_and_filter_chembl_data()
    
    # PASSO 2: Cria o ficheiro de tokens (.pt) e o tokenizer (.pkl) a partir do .txt
    if success_step1:
        create_tokenized_dataset()