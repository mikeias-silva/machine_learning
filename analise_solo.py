import pandas as pd
from io import StringIO
# ⚠️ CORREÇÃO: Importa todas as funções de regressão DIRETAMENTE do pycaret
from pycaret.regression import *

# ==============================================================================
# PARTE 1: DADOS E PRÉ-PROCESSAMENTO
# ==============================================================================

# Seus dados em formato de string.
dados_str = """Argila_g_kg-1	Silte_g_kg-1	AreiaTotal_g_kg-1	Ds_Mg_m3	Pt_m3_m3	Macro_m3_m3	Micro_m3_m3	CC_m3_m3	PMP_m3_m3	AD_m3_m3	CAD	CAS	RP_Mpa	DMP_mm	pHH20	H+Al_cmolc_dm-3	P_ mg_dm-3	K_cmolc_dm-3	Al_ cmolc_dm-3	Ca_cmolc_dm-3	Mg_cmolc_dm-3	CTCpH7_cmolc_dm-3	V%	COT_g_dm-3	CBM_µg C/g solo	RB_mg C de CO2 g-1 solo h-1	qCO2_mg C - CO2 g -1 BMS – C.h-1	GT_mg / g-1	GFE_mg / g-1	U_µg NH4-N g-1 seco h-1	FA _µg p-nitrofenol g-1 seco h-1	B_µg p-nitrofenol g-1 seco h-1	A _µg p-nitrofenol g-1 seco h-1	IQS (aditivo)
337,5	106,75	555,75	126,873,408	521,126,048	44,064,808	477,061,241	438,013,713	155,820,153	28,219,356	84,051,395	15,948,605	9,709	4,547,870,625	5,32	8,235,812,311	285,403,333	441,795,998	0,23	2,95	0,39	1,201,760,831	3,146,879,064	2,084,596,378	2,710,843,373	4,010,927,431	1,479,586,563	3,628,391,111	1,198,560,988	1,516,760,496	2,805,616,074	9,183,117,808	5,694,324,236	760,265,214
362,5	95,25	542,25	1,364,322,621	47,912,853	22,227,237	456,901,293	415,808,011	162,818,458	252,989,553	867,842,311	132,157,689	12,698	2,383,501,875	5,34	7,365,684,609	35,323,066	593,887,708	0,3	3,03	0,29	1,127,957,232	3,469,890,168	2,081,266,352	6,181,485,993	3,673,908,733	5,943,407,034	391,339,279	950,413,926	1,792,405,765	2,795,092,865	9,011,165,303	5,543,672,389	708,334,246
350	117	533	1,413,360,848	459,054,446	21,322,956	437,731,491	407,965,943	209,585,707	198,380,236	888,709,272	111,290,728	13,223	26,682,375	5,73	5,718,668,508	2,365,788,181	467,144,617	0,03	3,86	0,82	1,086,581,312	4,737,008,227	2,077,936,326	4,392,922,514	4,001,982,834	9,110,069,258	3,931,477,489	1,082,042,434	1,890,461,757	3,080,760,219	8,694,283,954	5,212,396,675	739,332,146
"""

# 1. Ler os dados como um DataFrame, usando tabulação como separador
df = pd.read_csv(StringIO(dados_str), sep='\t')

# 2. Corrigir o formato numérico
print("--- Iniciando Pré-processamento dos dados ---")
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

    # 2a. Substitui o separador de milhar (ponto) por nada (remove)
    # Regex ajustada para tentar pegar o separador de milhar
    df[col] = df[col].str.replace(r'(\d)\.(\d{3})', r'\1\2', regex=True)

    # 2b. Substitui a vírgula decimal (,) por ponto (.)
    df[col] = df[col].str.replace(',', '.', regex=False)

    # 2c. Converte para float (ignora erros)
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nPrimeiras linhas do DataFrame após a limpeza:")
print(df.head())
print("---------------------------------------------")

# ==============================================================================
# PARTE 2: CONFIGURAÇÃO E COMPARAÇÃO DE MODELOS COM PYCARET
# ==============================================================================

TARGET_VARIABLE = 'CBM_µg C/g solo'

print(f"\n--- 🎯 Configurando PyCaret para Regressão: Previsão de {TARGET_VARIABLE} ---\n")

# 1. Configurar o ambiente PyCaret
setup_pycaret = setup(data=df,
                      target=TARGET_VARIABLE,
                      session_id=42,  # Para reprodutibilidade
                      silent=True)

# 2. Comparar todos os modelos de regressão
print("--- 🚀 Comparando todos os modelos de Regressão ---")
melhor_modelo_comparado = compare_models()

# ==============================================================================
# PARTE 3: FOCO E ANÁLISE DO RANDOM FOREST
# ==============================================================================

print("\n--- 🌲 Foco no Random Forest Regressor (RF) ---")

# 3. Criar e treinar o modelo Random Forest
rf_base = create_model('rf')

# 4. Ajustar (Tunar) os hiperparâmetros do Random Forest para otimizar o desempenho
print("\n--- ⚙️ Otimizando Hiperparâmetros do Random Forest ---")
tuned_rf = tune_model(rf_base)

# 5. Avaliar a performance final e a importância das features
print("\n--- 📊 Avaliação de Performance e Importância das Features do RF Tunado ---")

plot_model(tuned_rf, plot='feature')
plot_model(tuned_rf, plot='error')
plot_model(tuned_rf, plot='predict')

# 6. Finalizar o modelo
final_rf = finalize_model(tuned_rf)

print("\n✅ Processo concluído! O modelo Random Forest otimizado está na variável 'final_rf'.")
print("Visualizações (Importância das Features e Gráficos de Erro) foram geradas.")