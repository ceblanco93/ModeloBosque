import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Leer los datos
data = pd.read_csv('archivo_convertido_nuevo.csv', parse_dates=['Fecha'])

# Obtener la lista única de productos
unique_products = data['ID del Producto'].unique()

# Agregar los datos por fecha y producto, sumando el stock final de todos los vendedores
data_aggregated_final = data.groupby(['Fecha', 'ID del Producto']).agg({'Stock Inicial': 'sum'}).reset_index()

# Crear lags para 'Stock Final'
for i in range(1, 13):
    data_aggregated_final[f'lag_{i}'] = data_aggregated_final.groupby('ID del Producto')['Stock Inicial'].shift(i)

# Eliminar las filas con NaN (debido a los lags)
data_aggregated_final = data_aggregated_final.dropna()

# Dividir las características y la variable objetivo
X_final = data_aggregated_final.drop(columns=['Fecha', 'Stock Inicial'])
y_final = data_aggregated_final['Stock Inicial']

# Entrenar el modelo de bosque aleatorio para 'Stock Final'
rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
rf_final.fit(X_final, y_final)

# Función para predecir los valores futuros para un producto específico y graficar los resultados
def predict_and_plot_for_product_final(product_id, model, train_data, product_name_dict):
    product_data = train_data[train_data['ID del Producto'] == product_id]
    if len(product_data) == 0:
        return
    last_known_data = product_data.iloc[-1]
    future_predictions = []
    future_dates = pd.date_range(train_data['Fecha'].max(), periods=7, freq='M')[1:]
    for _ in range(6):
        prediction = model.predict([last_known_data.drop(labels=['Fecha', 'Stock Inicial']).values])[0]
        future_predictions.append(prediction)
        for j in range(12, 1, -1):
            last_known_data[f'lag_{j}'] = last_known_data[f'lag_{j-1}']
        last_known_data['lag_1'] = prediction

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(product_data['Fecha'], product_data['Stock Inicial'], label='Datos historicos', color='blue')
    plt.plot(future_dates, future_predictions, label='Venta a 6 meses', color='red', linestyle='--', marker='o')
    plt.title(f'Stock Inicial predicciones de  {product_name_dict[product_id]}')
    plt.xlabel('Fecha')
    plt.ylabel('Stock Inicial')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Generar predicciones y gráficos para 'Stock Final'
product_name_dict = data.set_index('ID del Producto')['Nombre del Producto'].to_dict()
for product_id in unique_products:
    predict_and_plot_for_product_final(product_id, rf_final, data_aggregated_final, product_name_dict)
