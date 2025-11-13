import pandas as pd
import numpy as np
# ⚠️ CORREÇÃO: Importa todas as funções de regressão DIRETAMENTE do pycaret
from pycaret.regression import *

# ==============================================================================
# PARTE 1: DADOS E PRÉ-PROCESSAMENTO
# ==============================================================================

print("--- 🔄 Carregando dados do arquivo database.csv ---")
# 1. Ler os dados completos do arquivo CSV (separador: ponto e vírgula)
df = pd.read_csv('database.csv', sep=';')

# 1.1 Remover espaços em branco dos nomes das colunas
df.columns = df.columns.str.strip()

# 2. Corrigir o formato numérico (vírgula para ponto decimal)
print("--- Iniciando Pré-processamento dos dados ---")
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()
    # Substitui a vírgula decimal (,) por ponto (.)
    df[col] = df[col].str.replace(',', '.', regex=False)
    # Converte para float
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remover linhas com valores faltantes
print(f"\n📊 Shape original: {df.shape}")
df = df.dropna()
print(f"📊 Shape após remover NaN: {df.shape}")

print("\n✅ Primeiras linhas do DataFrame após limpeza:")
print(df.head())
print("\n📈 Estatísticas descritivas da variável alvo:")
print(df['CBM_µg C/g solo'].describe())
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
                      fold=5,  # Validação cruzada com 5 folds
                      normalize=True,  # Normalizar features
                      transformation=False,  # Sem transformação de alvo
                      verbose=False)

# 2. Comparar todos os modelos de regressão
print("\n--- 🚀 Comparando TODOS os modelos de Regressão disponíveis ---")
print("Métricas: MAE, MSE, RMSE, R², RMSLE, MAPE")
print("=" * 80)
melhor_modelo_comparado = compare_models(sort='R2', n_select=3)  # Top 3 modelos
print("\n✅ Comparação completa! Os 3 melhores modelos foram selecionados.")

# ==============================================================================
# PARTE 3: FOCO E ANÁLISE DO RANDOM FOREST
# ==============================================================================

print("\n" + "=" * 80)
print("--- 🌲 ANÁLISE DETALHADA: Random Forest Regressor (RF) ---")
print("=" * 80)

# 3. Criar e treinar o modelo Random Forest
print("\n📦 Criando modelo Random Forest base...")
rf_base = create_model('rf', verbose=False)
print("✅ Random Forest base criado!")

# 4. Ajustar (Tunar) os hiperparâmetros do Random Forest para otimizar o desempenho
print("\n--- ⚙️ Otimizando Hiperparâmetros do Random Forest ---")
print("Isso pode levar alguns minutos...")
tuned_rf = tune_model(rf_base, optimize='R2', n_iter=10, verbose=False)
print("✅ Otimização concluída!")

# 5. Avaliar a performance final e a importância das features
print("\n--- 📊 Avaliação de Performance e Importância das Features do RF Tunado ---")

print("\n📈 Gerando gráfico de Importância das Features...")
plot_model(tuned_rf, plot='feature', save=True)

print("📉 Gerando gráfico de Análise de Resíduos...")
plot_model(tuned_rf, plot='residuals', save=True)

print("🎯 Gerando gráfico de Erro de Predição...")
plot_model(tuned_rf, plot='error', save=True)

# 6. Finalizar o modelo
final_rf = finalize_model(tuned_rf)

# ==============================================================================
# PARTE 4: COMPARAÇÃO DETALHADA - RANDOM FOREST VS OUTROS MODELOS
# ==============================================================================

print("\n" + "=" * 80)
print("--- 📊 COMPARAÇÃO: Random Forest vs Outros Modelos Top ---")
print("=" * 80)

# Criar modelos adicionais para comparação
print("\n🔬 Criando modelos adicionais para comparação...")

models_to_compare = {
    'Random Forest': tuned_rf,
    'Extra Trees': create_model('et', verbose=False),
    'Gradient Boosting': create_model('gbr', verbose=False),
    'LightGBM': create_model('lightgbm', verbose=False),
    'XGBoost': create_model('xgboost', verbose=False)
}

print("\n📋 Resumo de Performance dos Modelos:")
print("=" * 80)

# Avaliar todos os modelos
for name, model in models_to_compare.items():
    print(f"\n🔹 {name}:")
    metrics = pull()  # Obtém as métricas do último modelo criado/tunado
    if not metrics.empty:
        print(f"   R² (Mean): {metrics['R2'].mean():.4f}")
        print(f"   MAE (Mean): {metrics['MAE'].mean():.4f}")
        print(f"   RMSE (Mean): {metrics['RMSE'].mean():.4f}")

print("\n" + "=" * 80)
print("✅ PROCESSO COMPLETO!")
print("=" * 80)
print("\n📁 Arquivos gerados:")
print("   • Feature Importance (Feature Importance.png)")
print("   • Residuals Plot (Residuals.png)")
print("   • Prediction Error (Prediction Error.png)")
print("\n🎯 Modelo Random Forest otimizado está na variável 'final_rf'")
print("📊 Use 'predict_model(final_rf, data=novos_dados)' para fazer previsões\n")