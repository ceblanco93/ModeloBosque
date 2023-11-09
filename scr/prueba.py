import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Leer los datos
data = pd.read_csv('archivo_convertido_nuevo.csv', parse_dates=['Fecha'])

# Crear columnas de año, mes y día
data['Year'] = data['Fecha'].dt.year
data['Month'] = data['Fecha'].dt.month
data['Day'] = data['Fecha'].dt.day

# Agregar los datos por fecha y producto
data_aggregated = data.groupby(['Fecha', 'ID del Producto', 'Year', 'Month', 'Day']).agg({'Stock Inicial': 'sum'}).reset_index()

# Crear lags
for i in range(1, 7):
    data_aggregated[f'lag_{i}'] = data_aggregated.groupby('ID del Producto')['Stock Inicial'].shift(i)

# Eliminar las filas con espacios nulos esdecir que contenga un valor cero
data_aggregated = data_aggregated.dropna()

# Dividir las características y la variable objetivo para el entrenamiento
X_train = data_aggregated[['Year', 'Month', 'Day', 'ID del Producto'] + [f'lag_{i}' for i in range(1, 7)]]
y_train = data_aggregated['Stock Inicial']

# Entrenar el regresor de bosque aleatorio en los datos agregados
rf_aggregated = RandomForestRegressor(n_estimators=100, random_state=42)
rf_aggregated.fit(X_train, y_train)

# Función para predecir para un producto específico con datos agregados para un período específico
def predict_for_product_aggregated_period(product_id, model, train_data, start_month, end_month):
    product_data = train_data[train_data['ID del Producto'] == product_id]
    if len(product_data) == 0:
        return [np.nan] * (end_month - start_month)
    last_known_data = product_data.iloc[-1]
    future_predictions = []
    for i in range(start_month, end_month):
        last_known_data['Month'] = i
        if i == 1:
            last_known_data['Year'] += 1
        prediction = model.predict([last_known_data[['Year', 'Month', 'Day', 'ID del Producto'] + [f'lag_{i}' for i in range(1, 7)]].values])[0]
        future_predictions.append(prediction)
        # Actualizar los valores de lag para la siguiente predicción
        for j in range(6, 0, -1):
            last_known_data[f'lag_{j}'] = last_known_data[f'lag_{j-1}'] if j > 1 else prediction
    return future_predictions

# Generar predicciones y graficar 
unique_products = data['ID del Producto'].unique()
product_name_dict = data.set_index('ID del Producto')['Nombre del Producto'].to_dict()
for product_id in unique_products:
    product_data = data_aggregated[data_aggregated['ID del Producto'] == product_id]
    dates = product_data['Fecha']
    real_values = product_data['Stock Inicial']
    predictions_aggregated_period = predict_for_product_aggregated_period(product_id, rf_aggregated, data_aggregated, 6, 12)
    future_dates = [pd.Timestamp(2023, i, 1) for i in range(6, 12)]
    plt.plot(dates, real_values, label='Valores Históricos', color='blue')
    plt.plot(future_dates, predictions_aggregated_period, label='Predicciones', color='red', linestyle='--', marker='o')
    plt.xlabel('Fecha')
    plt.ylabel('Stock Inicial')
    plt.title(f'Producto: {product_name_dict[product_id]}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
