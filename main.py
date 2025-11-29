import pandas as pd

df = pd.read_excel("investimentos---tt.xlsx")
colunas_uf = [c for c in df.columns if c.startswith("UF PT-")]

colunas_importantes = ["Ano Lançamento", "Movim. Líquido - R$ (Item Informação)"] + colunas_uf

df = df[colunas_importantes]

# Converter Movimento Líquido
df["Movim. Líquido - R$ (Item Informação)"] = (
    df["Movim. Líquido - R$ (Item Informação)"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

X = df.drop("Movim. Líquido - R$ (Item Informação)", axis=1)
y = df["Movim. Líquido - R$ (Item Informação)"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("Treinado!")



#Métricas quantitativas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = modelo.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R²:", r2)
print("MAE:", mae)
print("RMSE:", rmse)



#Mapa de correlação
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(), annot=False)
plt.title("Mapa de Correlação")
plt.show()

#Dados reais x Dados previstos
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")
plt.title("Valores Reais vs Valores Previstos")
plt.grid(True)
plt.show()


#Resíduos
import matplotlib.pyplot as plt

residuos = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuos)
plt.axhline(y=0, linestyle='--')
plt.xlabel("Valores Previstos")
plt.ylabel("Resíduos (Erro)")
plt.title("Gráfico de Resíduos")
plt.grid(True)
plt.show()


