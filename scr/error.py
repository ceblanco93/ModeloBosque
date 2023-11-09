import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Carga de datos
data = pd.read_csv('archivo_convertido_nuevo.csv')
data['Fecha'] = pd.to_datetime(data['Fecha'])
data = data.sort_values(by=['ID del Producto', 'Fecha'])

# Creación de características basadas en tiempo
data['Year'] = data['Fecha'].dt.year
data['Month'] = data['Fecha'].dt.month
data['Day'] = data['Fecha'].dt.day

# Codificación de columnas categóricas
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for column in categorical_columns:
    if column != 'Fecha':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Limpiar los valores que contenga valores nulos o ceros de la variable objetivo
data_clean = data.dropna(subset=['Stock Inicial'])

# Definir características y variable objetivo
features = [col for col in data_clean.columns if col not in ['Stock Inicial', 'Fecha', 'Stock Final','Costo Unitario','Costo por Unidad de Producto']]  # Ajusta los nombres de las columnas
X = data_clean[features]
y = data_clean['Stock Inicial']

# Imputación de valores faltantes
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicción y cálculo de métricas
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X_test, y_test)

# Mostrar métricas
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

# Gráfica de valores reales vs. predicciones
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Valores reales')
plt.ylabel('Valores de la prediccion')
plt.title('Valores reales vs Predicion')
plt.show()
