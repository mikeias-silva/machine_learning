# Importar bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Criar um DataFrame simples
data = {
        'tamanho_m2':       [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        'preco_k_reais':    [150, 180, 200, 230, 250, 280, 310, 340, 370, 400]
        }
df = pd.DataFrame(data)

# Separar a variável de entrada (preço e tamanho) e a variável de saída (preço)
X = df[['preco_k_reais']]
y = df['tamanho_m2']

# Dividir os dados em treino e teste (70% para treino, 30% para teste)
X_train_tamanho, X_test_tamanho, y_train_preco, y_test_preco = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo de Regressão Linear
model_tamanho = LinearRegression()
model_tamanho.fit(X_train_tamanho, y_train_preco)

# Fazer previsões com os dados de teste
y_pred_tamanho = model_tamanho.predict(X_test_tamanho)
mse = mean_squared_error(y_test_preco, y_pred_tamanho)
print(f"Erro Quadrático Médio (MSE) para prever o: {mse:.2f}")

# --- Nova parte do código: Prever e plotar novos pontos ---

# Novos tamanhos para previsão
novos_precos = pd.DataFrame({'preco_k_reais': [200, 230.1, 333]})
previsoes_novos_precos = model_tamanho.predict(novos_precos)

# Visualizar a linha de regressão e os novos pontos
plt.figure(figsize=(10, 6)) # Aumenta o tamanho do gráfico para melhor visualização

# Plotar os dados reais
plt.scatter(X, y, color='blue', label='Dados Reais')

# Plotar a linha de regressão
plt.plot(X, model_tamanho.predict(X), color='red', label='Linha de Regressão')

# Plotar os novos pontos previstos
# Use um marcador diferente para destacá-los
plt.scatter(novos_precos, previsoes_novos_precos, color='green', marker='X', s=100, label='Novas Previsões')

plt.title('Regressão Linear: Linha de Previsão de tamanho com Base no Preço')
plt.xlabel('Preço em R$ (Milhares)')
plt.ylabel('Tamanho em m²')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir as previsões no console
print("\n--- Previsões de tamanho para novos preços ---")
print(f"Tamanho previsto para uma casa de R$ 550 mil: {previsoes_novos_precos[0]:.2f} m²")
print(f"Tamanho previsto para uma casa de R$ 700 mil:  {previsoes_novos_precos[1]:.2f} m²")
print(f"Tamanho previsto para uma casa de R$ 900 mil:  {previsoes_novos_tamanhos[2]:.2f} m²")