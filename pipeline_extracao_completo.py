# pipeline_extracao_completo.py (Versão para múltiplos arquivos)
import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURAÇÕES E MAPEAMENTOS ---
MUNICIPIO_JF = 313670

# *** ALTERAÇÃO PRINCIPAL: LISTA DE ARQUIVOS A SEREM PROCESSADOS ***
# O script irá iterar sobre esta lista.
BASE_DIR = r"D:\2025\UFJF\ETLSIH"
ARQUIVOS_DADOS = [
    "ETLSIH.ST_MG_2025_5_t.csv",
    "ETLSIH.ST_MG_2025_4_t.csv",
    "ETLSIH.ST_MG_2025_3_t.csv",
    "ETLSIH.ST_MG_2025_2_t.csv",
    "ETLSIH.ST_MG_2025_1_t.csv",
    "ETLSIH.ST_MG_2024_12_t.csv",
    "ETLSIH.ST_MG_2024_11_t.csv",
    "ETLSIH.ST_MG_2024_10_t.csv",
    "ETLSIH.ST_MG_2024_9_t.csv",
    "ETLSIH.ST_MG_2024_8_t.csv",
    "ETLSIH.ST_MG_2024_7_t.csv",
    "ETLSIH.ST_MG_2024_6_t.csv",
    "ETLSIH.ST_MG_2024_5_t.csv",
    "ETLSIH.ST_MG_2024_4_t.csv",
    "ETLSIH.ST_MG_2024_3_t.csv",
    "ETLSIH.ST_MG_2024_2_t.csv",
    "ETLSIH.ST_MG_2024_1_t.csv",
    "ETLSIH.ST_MG_2023_12_t.csv",
    "ETLSIH.ST_MG_2023_11_t.csv",
    "ETLSIH.ST_MG_2023_10_t.csv",
    "ETLSIH.ST_MG_2023_9_t.csv",
    "ETLSIH.ST_MG_2023_8_t.csv",
    "ETLSIH.ST_MG_2023_7_t.csv",
    "ETLSIH.ST_MG_2023_6_t.csv",
    "ETLSIH.ST_MG_2023_5_t.csv",
    "ETLSIH.ST_MG_2023_4_t.csv",
    "ETLSIH.ST_MG_2023_3_t.csv",
    "ETLSIH.ST_MG_2023_2_t.csv",
    "ETLSIH.ST_MG_2023_1_t.csv",
    "ETLSIH.ST_MG_2022_12_t.csv",
    "ETLSIH.ST_MG_2022_11_t.csv",
    "ETLSIH.ST_MG_2022_10_t.csv",
    "ETLSIH.ST_MG_2022_9_t.csv",
    "ETLSIH.ST_MG_2022_8_t.csv",
    "ETLSIH.ST_MG_2022_7_t.csv",
    "ETLSIH.ST_MG_2022_6_t.csv",
    "ETLSIH.ST_MG_2022_5_t.csv",
]

# Mapeamento de CIDs para categorias de risco (pode ser expandido)
CIDS_CRITICOS = ['I21', 'I60', 'I61', 'I63', 'A41', 'J80', 'S06'] # Infarto, AVC, Sepse, SARA, Trauma Craniano Grave
CIDS_URGENTES = ['J18', 'N17', 'K72', 'C', 'I50'] # Pneumonia, Insuf. Renal Aguda, Cânceres, Insuf. Cardíaca

# --- 2. FUNÇÕES AUXILIARES ---
def carregar_multiplos_arquivos(base_dir, arquivos):
    """Carrega e concatena múltiplos arquivos CSV do DATASUS."""
    lista_dfs = []
    print("🚀 Iniciando carregamento de múltiplos arquivos...")
    for nome_arquivo in arquivos:
        caminho_completo = os.path.join(base_dir, nome_arquivo)
        print(f"  -> Processando arquivo: {nome_arquivo}")
        if os.path.exists(caminho_completo):
            try:
                df = pd.read_csv(caminho_completo, encoding='latin-1', low_memory=False, sep=',')
                lista_dfs.append(df)
                print(f"     ✅ Carregado com {len(df):,} linhas.")
            except Exception as e:
                print(f"     ⚠️  AVISO: Falha ao ler o arquivo {nome_arquivo}. Erro: {e}")
        else:
            print(f"     ❌ ERRO: Arquivo não encontrado em {caminho_completo}.")
    
    if not lista_dfs:
        print("❌ Nenhum arquivo foi carregado. Encerrando.")
        return None
        
    df_total = pd.concat(lista_dfs, ignore_index=True)
    print(f"\n✅ Concatenação concluída! Total de {len(df_total):,} registros carregados.")
    return df_total

# As outras funções permanecem praticamente as mesmas
def filtrar_dados_jf(df):
    print(f"\n🔍 Filtrando registros para Juiz de Fora (MUNIC_MOV = {MUNICIPIO_JF})...")
    df_jf = df[df['MUNIC_MOV'] == MUNICIPIO_JF].copy()
    print(f"🏥 Encontrados {len(df_jf):,} registros em Juiz de Fora no período total.")
    return df_jf

def criar_features_relevantes(df):
    print("\n🛠️  Criando e limpando features...")
    
    # Datas
    df['DT_INTER'] = pd.to_numeric(df['DT_INTER'], errors='coerce')
    df['DT_SAIDA'] = pd.to_numeric(df['DT_SAIDA'], errors='coerce')
    df = df.dropna(subset=['DT_INTER', 'DT_SAIDA'])
    df['DT_INTER'] = pd.to_datetime(df['DT_INTER'], format='%Y%m%d', errors='coerce')
    df['DT_SAIDA'] = pd.to_datetime(df['DT_SAIDA'], format='%Y%m%d', errors='coerce')
    
    # Tempo de permanência
    df['LOS_DIAS'] = (df['DT_SAIDA'] - df['DT_INTER']).dt.days
    df = df[(df['LOS_DIAS'] >= 0) & (df['LOS_DIAS'] < 365)] 
    
    # Features numéricas e categóricas
    df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')
    df['DIAG_PRINC'] = df['DIAG_PRINC'].astype(str).str.strip()
    df['CAR_INT'] = df['CAR_INT'].astype(str)
    
    return df

def aplicar_filtro_relevancia_uti(df):
    print("\n🔬 Aplicando Filtro de Relevância Clínica para UTI...")
    
    # Garante que as colunas existam antes de usar
    required_cols = ['MARCA_UTI', 'MARCA_UCI', 'DIAG_PRINC', 'LOS_DIAS', 'MORTE', 'PROC_REA']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 # Adiciona a coluna com valor padrão se não existir
            
    cond_uti_explicita = (df['MARCA_UTI'].isin(['1', '2', '3', 1, 2, 3])) | (df['MARCA_UCI'].isin(['1', 1]))
    cond_diag_critico = df['DIAG_PRINC'].str[:3].isin(CIDS_CRITICOS)
    # Câncer (qualquer CID começando com C)
    cond_cancer = df['DIAG_PRINC'].str.startswith('C')
    cond_diag_urgente = df['DIAG_PRINC'].str[:3].isin(CIDS_URGENTES)
    cond_obito = df['MORTE'] == 1
    cond_proc_complexo = df['PROC_REA'].astype(str).str.startswith(('04', '05', '06', '07')) # Cirúrgicos, Transplantes, etc

    df_candidatos_uti = df[
        cond_uti_explicita | 
        cond_diag_critico | 
        cond_cancer |
        cond_diag_urgente |
        cond_obito |
        cond_proc_complexo
    ].copy()
    
    print(f"📊 População de interesse para UTI reduzida para {len(df_candidatos_uti):,} pacientes.")
    return df_candidatos_uti

def classificar_prioridade(df):
    print("⚖️  Classificando prioridade dos pacientes...")
    
    def get_prioridade(row):
        diag = str(row['DIAG_PRINC'])[:3]
        car_int = str(row['CAR_INT'])
        if diag in CIDS_CRITICOS or car_int == '02': return 1
        elif diag in CIDS_URGENTES or car_int == '03': return 2
        elif car_int == '01': return 4
        else: return 3
            
    df['gravidade_gi'] = df.apply(get_prioridade, axis=1)
    return df

def main():
    """Pipeline principal de extração e preparação de dados."""
    
    # --- ETAPA 1: CARREGAMENTO E FILTRAGEM INICIAL ---
    df_total_raw = carregar_multiplos_arquivos(BASE_DIR, ARQUIVOS_DADOS)
    if df_total_raw is None: return
    
    df_jf_raw = filtrar_dados_jf(df_total_raw)
    
    # --- ETAPA 2: PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES ---
    df_jf_processed = criar_features_relevantes(df_jf_raw)
    
    # --- ETAPA 3: DEFINIÇÃO DA COORTE DE INTERESSE ---
    df_uti_candidates = aplicar_filtro_relevancia_uti(df_jf_processed)
    
    # --- ETAPA 4: ENRIQUECIMENTO DOS DADOS PARA OTIMIZAÇÃO ---
    df_final = classificar_prioridade(df_uti_candidates)
    
    # --- ETAPA 5: SEPARAÇÃO E EXPORTAÇÃO DOS DATASETS ---
    print("\n💾 Exportando datasets para as próximas etapas...")

    # Adicionar ID único
    df_final = df_final.reset_index(drop=True).reset_index().rename(columns={'index': 'paciente_id'})
    
    # Dataset para Machine Learning
    colunas_ml = ['paciente_id', 'IDADE', 'SEXO', 'DIAG_PRINC', 'CAR_INT', 'PROC_REA', 'MORTE', 'LOS_DIAS']
    df_ml = df_final[colunas_ml].dropna()
    df_ml.to_csv('dataset_para_ml.csv', index=False)
    print(f"   ✅ 'dataset_para_ml.csv' criado com {len(df_ml):,} registros.")

    # Dataset para Otimização
    df_otimizacao = df_final[['paciente_id', 'DT_INTER', 'gravidade_gi', 'LOS_DIAS']].copy()
    df_otimizacao.rename(columns={'DT_INTER': 'tempo_chegada_ci', 'LOS_DIAS': 'duracao_di_real'}, inplace=True)
    df_otimizacao = df_otimizacao[['paciente_id', 'tempo_chegada_ci', 'gravidade_gi', 'duracao_di_real']]
    
    df_otimizacao.to_csv('dataset_para_otimizacao.csv', index=False)
    print(f"   ✅ 'dataset_para_otimizacao.csv' criado com {len(df_otimizacao):,} registros.")

    print("\n🎉 Processo concluído com sucesso!")
    print("\n👉 PRÓXIMO PASSO: Execute o script 'gerador_de_resultados.py' para treinar o modelo de ML com este dataset maior e mais robusto.")

if __name__ == "__main__":
    main()