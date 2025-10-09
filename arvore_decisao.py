import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Criando o DataFrame simulado
data = {
    'Renda':             [6000, 3000, 4500, 7000, 5500, 3500, 8000, 4000],
    'Historico_Credito': [1, 0, 1, 1, 0, 1, 1, 0], # 1=Bom, 0=Ruim
    'Garantia':          [1, 0, 1, 1, 0, 0, 1, 0], # 1=Sim, 0=Não
    'Aprova':            [1, 0, 1, 1, 0, 0, 1, 0] # 1=Aprovado, 0=Rejeitado
}

df = pd.DataFrame(data)

# definindo X ( Variaves de entrada) e y (variável de saída)
X = df[['Renda', 'Historico_Credito', 'Garantia']]
y = df['Aprova']

# 2. Dividir para TReinamento e Teste
# Dividir os dados (mesmo que seja um dataset pequeno, é a prática padrão)
(X_train,
 X_test,
 y_train,
 y_test) = (
    train_test_split(
        X,
                y,
                test_size=0.3,
                random_state=42
))
print("Dados de treino prontos para o modelo!")

# 3. Criar e treinar o modelo
tree_model = DecisionTreeClassifier(random_state=42)

# Treinar o modelo
tree_model.fit(X_train, y_train)
print("Modelo de Árvore de Decisão treinado!")

# 4. Fazer uma previsão e interpretar (A saída)
y_pred = tree_model.predict(X_test)

print("Previsões (0=Rejeitado, 1=Aprovado):")
print(y_pred)
#saída: [0 1 1] Depende da aleatoriedade do split

#Acurácia (Quão bem o modelo se saiu no teste)
acuracia = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do Modelo: {acuracia*100:.2f}%")

novo_pedido = pd.DataFrame([[6000, 1, 1]], columns=['Renda', 'Historico_Credito', 'Garantia'])
previsao = tree_model.predict(novo_pedido)
print("Aprovado pela arvore" if previsao[0] == 1 else "Rejeitado pela arvore")

###Random Forest
#1. criar e treinar o modelo random forest
# O parmaetro n_estimators define o numero de arvores na floresta (vamos usar 100)
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

#2. Treinar o modelo (ele treinará 100 ãrvores)
forest_model.fit(X_train, y_train)

print("Modelo random forest treinado com 100 arvores!")

#Fazer previsões usando a FLORESTA
y_pred_forest = forest_model.predict(X_test)

#comaprar a performance
acuracia_forest = accuracy_score(y_test, y_pred_forest)

print("\n---- Resultados do Teste (3 clientes)----")
print(f"Previsões do Random forest: {y_pred_forest}")
print(f"Resultados Reais (y_test): [0 1 1]")

print(f"\nAcurácia da Unica árvore de decisão: {acuracia*100:.2f}%")
print(f"Acurácia do RANDOM FOREST: {acuracia_forest*100:.2f}%")


previsao_forest = forest_model.predict(novo_pedido)
print("Aprovado pela arvore" if previsao[0] == 1 else "Rejeitado pela arvore")
print("Aprovado pelo random forest" if previsao_forest[0] == 1 else "Rejeitado pelo random forest")

# Previsão da Árvore de Decisão
previsao_tree = tree_model.predict(novo_pedido)[0]
# Previsão do Random Forest
previsao_forest = forest_model.predict(novo_pedido)[0]

# Previsão detalhada da VOTAÇÃO (para explicação)
# O .predict_proba dá a probabilidade (a "votação") para 0 e 1
votos = forest_model.predict_proba(novo_pedido)[0]

# --- IMPRESSÃO DOS RESULTADOS E MOTIVOS ---

print("="*50)
print("ANÁLISE DO PEDIDO (Renda 6000, Histórico Bom, Garantia Sim)")
print("Resultado Verdadeiro: APROVADO (1)")
print("="*50)

# 1. Resultado da Árvore de Decisão
print("1. ÁRVORE DE DECISÃO (DecisionTreeClassifier):")
print(f"   -> Previsão: {'APROVADO (1)' if previsao_tree == 1 else 'REJEITADO (0)'}")

# 2. Resultado do Random Forest
print("\n2. RANDOM FOREST (RandomForestClassifier):")
print(f"   -> Previsão: {'APROVADO (1)' if previsao_forest == 1 else 'REJEITADO (0)'}")
print("="*50)


# --- 1. TREINAMENTO DOS MODELOS (Usando random_state=10 para mudar a regra da Árvore) ---
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.3, random_state=10) # Mudando a aleatoriedade

# Treinar a Árvore de Decisão Simples com nova regra
tree_model_fail = DecisionTreeClassifier(random_state=42).fit(X_train_new, y_train_new)

# Treinar o Random Forest (100 árvores)
forest_model_stable = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_new, y_train_new)

# --- 2. O PEDIDO DE TESTE (Renda 5500, Histórico Ruim, Garantia Não) ---
pedido_fraco = pd.DataFrame([[1, 1, 1]], columns=['Renda', 'Historico_Credito', 'Garantia'])

previsao_tree_fail = tree_model_fail.predict(pedido_fraco)[0]
previsao_forest_stable = forest_model_stable.predict(pedido_fraco)[0]

votos_forest_stable = forest_model_stable.predict_proba(pedido_fraco)[0]

print("="*70)
print("ANÁLISE DO CLIENTE: Renda 5500, Histórico RUIM, Garantia NÃO")
print("Resultado VERDADEIRO: REJEITADO (0)")
print("="*70)

# 1. ÁRVORE DE DECISÃO (O Juiz Solitário - Agora Falhando)
print("\n[DECISION TREE CLASSIFIER - ÁRVORE ÚNICA]")
previsao_tree_str = "APROVADO (1)" if previsao_tree_fail == 1 else "REJEITADO (0)"
print(f"-> DECISÃO: {previsao_tree_str}")
if previsao_tree_fail == 1:
    print("-> MOTIVO DA FALHA:")
    print("   A Árvore de Decisão, com uma nova regra (devido ao `random_state` diferente), pode ter superestimado a Renda (5500) ou a Garantia, gerando uma regra que leva à **Aprovação Incorreta** (falso positivo).")

# 2. RANDOM FOREST (O Júri Coletivo - Mais Cauteloso)
print("\n[RANDOM FOREST CLASSIFIER - FLORESTA DE ÁRVORES]")
previsao_forest_str = "APROVADO (1)" if previsao_forest_stable == 1 else "REJEITADO (0)"
print(f"-> DECISÃO: {previsao_forest_str}")
print("-> VOTAÇÃO (0=Rejeitar, 1=Aprovar):")
print(f"   {votos_forest_stable[0]*100:.1f}% para Rejeitar (0) | {votos_forest_stable[1]*100:.1f}% para Aprovar (1)")
if previsao_forest_stable == 0:
    print("-> MOTIVO DO ACERTO:")
    print("   O Random Forest usa 100 árvores. A maioria (votos para 0) reconheceu o risco do Histórico Ruim e da falta de Garantia, ignorando o valor médio da Renda. **O consenso coletivo evita o erro da árvore solitária.**")
print("="*70)

datasolo = {
    'Argila_g_kg-1': [337.5, 362.5],
    'Silte_g_kg-1': [106.75, 95.25],
    'AreiaTotal_g_kg-1': [555.75, 542.25],
    'Ds_Mg_m3': [1.26873408, 1.364322621],
    'Pt_m3_m3': [0.521126048, 0.47912853],
    'Macro_m3_m3': [0.044064808, 0.022227237],
    'Micro_m3_m3': [0.477061241, 0.456901293],
    'pHH20': [5.32, 5.34],
    'COT_g_dm-3': [31.46879064, 34.69890168],
    'CBM_µg C/g solo': [20.84596378, 20.81266352],
    'RB_mg C de CO2 g-1 solo h-1': [27.10843373, 61.81485993],
    'qCO2_mg C - CO2 g -1 BMS – C.h-1': [4.010927431, 3.673908733],
    'IQS (aditivo)': [0.760265214, 0.708334246]
}

df = pd.DataFrame(datasolo)

# --- Exemplo de separação entre entradas e saídas ---
# X = físicos e químicos (sem os biológicos)
X = df[['Argila_g_kg-1', 'Silte_g_kg-1', 'AreiaTotal_g_kg-1',
        'Ds_Mg_m3', 'Pt_m3_m3', 'Macro_m3_m3', 'Micro_m3_m3', 'pHH20']]

# y = atributos biológicos (exemplo)
y = df[['COT_g_dm-3', 'CBM_µg C/g solo', 'RB_mg C de CO2 g-1 solo h-1',
        'qCO2_mg C - CO2 g -1 BMS – C.h-1', 'IQS (aditivo)']]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Criação e treino do modelo Random Forest (para regressão)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)



# Previsão
y_pred = model.predict(X_test)

# Avaliação
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))


# Exemplo de previsão para novo solo
novo_solo = pd.DataFrame([[350, 100, 550, 1.3, 0.5, 0.03, 0.47, 5.4]],
                         columns=X.columns)
previsao = model.predict(novo_solo)
print("\nPrevisão de atributos biológicos:")
print(previsao)