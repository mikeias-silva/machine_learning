import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
data = pd.read_csv("database.csv", sep=";")
data.columns = [col.strip() for col in data.columns]

# Verificar nomes das colunas
print(data.columns)

# Separar X e y
X = data.drop('IQS (aditivo)', axis=1)
y = data['IQS (aditivo)']

# Converter todos os dados para float (caso tenha vírgula como decimal)
X = X.apply(lambda x: x.str.replace(',', '.').astype(float) if x.dtype == 'object' else x)
y = y.apply(lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else x)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Previsão
y_pred = model.predict(X_test)

# Avaliar (sem 'squared')
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R²:", "indefinido (apenas 1 amostra de teste)" if len(y_test) < 2 else r2_score(y_test, y_pred))
print("RMSE:", rmse)

# Novo solo (somente físicos e químicos)
novo_solo = pd.DataFrame([[
    350, 100, 550, 1.3, 0.5, 0.03, 0.47, 0.438, 0.156, 0.282, 0.841, 0.159, 0.971,
    4.55, 5.3, 8.24, 28.54, 0.442, 0.23, 2.95, 0.39, 12.02, 31.47
]], columns=X.columns)

# Previsão dos atributos biológicos
previsao = model.predict(novo_solo)
print("Previsão dos atributos biológicos do solo:")
print(previsao)
