# Importar bibliotecas
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