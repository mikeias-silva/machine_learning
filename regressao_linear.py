import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Criar um DataFrame simples
data = {'tamanho_m2': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        'preco_k_reais': [150, 180, 200, 230, 250, 280, 310, 340, 370, 400]}
df = pd.DataFrame(data)

# Visualizar os dados
plt.scatter(df['tamanho_m2'], df['preco_k_reais'])
plt.title('Preço da Casa vs. Tamanho')
plt.xlabel('Tamanho em m²')
plt.ylabel('Preço em R$ (Milhares)')
plt.show()

# Separar a variável de entrada (tamanho) e a variável de saída (preço)
X = df[['tamanho_m2']]
y = df['preco_k_reais']

# Dividir os dados em treino e teste (70% para treino, 30% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo de Regressão Linear
model = LinearRegression()

# Treinar o modelo com os dados de treino
model.fit(X_train, y_train)

# Criar o modelo de Regressão Linear
model = LinearRegression()

# Treinar o modelo com os dados de treino
model.fit(X_train, y_train)


# Fazer previsões com os dados de teste
y_pred = model.predict(X_test)

# Avaliar o modelo usando o Erro Quadrático Médio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")

# Visualizar a linha de regressão
plt.scatter(X, y, label='Dados Reais')
plt.plot(X, model.predict(X), color='red', label='Linha de Regressão')
plt.title('Regressão Linear: Linha de Previsão')
plt.xlabel('Tamanho em m²')
plt.ylabel('Preço em R$ (Milhares)')
plt.legend()
plt.show()

# Crie uma lista com os novos tamanhos para previsão
novos_tamanhos = pd.DataFrame({'tamanho_m2': [550, 700, 900]})
# Fazer previsões para os novos tamanhos
