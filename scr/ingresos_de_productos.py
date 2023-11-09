import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import matplotlib.ticker as ticker

# Leer los datos
data = pd.read_csv('archivo_convertido_nuevo.csv', parse_dates=['Fecha'])

# Obtener la lista única de productos
unique_products = data['ID del Producto'].unique()

# Agregar los datos por fecha y producto, sumando los ingresos de todos los vendedores
data_aggregated_revenue = data.groupby(['Fecha', 'ID del Producto']).agg({'Ingresos de Productos': 'sum'}).reset_index()

# Crear lags para Ingresos de Productos
for i in range(1, 13):
    data_aggregated_revenue[f'lag_{i}'] = data_aggregated_revenue.groupby('ID del Producto')['Ingresos de Productos'].shift(i)

# Eliminar las filas con valores cero
data_aggregated_revenue = data_aggregated_revenue.dropna()

# Dividir las características y la variable objetivo
X_revenue = data_aggregated_revenue.drop(columns=['Fecha', 'Ingresos de Productos'])
y_revenue = data_aggregated_revenue['Ingresos de Productos']

# Entrenar el modelo de bosque aleatorio 
rf_revenue = RandomForestRegressor(n_estimators=100, random_state=42)
rf_revenue.fit(X_revenue, y_revenue)

# Función para predecir los valores futuros
def predict_and_plot_for_product_revenue(product_id, model, train_data, product_name_dict):
    product_data = train_data[train_data['ID del Producto'] == product_id]
    if len(product_data) == 0:
        return
    last_known_data = product_data.iloc[-1]
    future_predictions = []
    future_dates = pd.date_range(train_data['Fecha'].max(), periods=7, freq='M')[1:]
    for _ in range(6):
        prediction = model.predict([last_known_data.drop(labels=['Fecha', 'Ingresos de Productos']).values])[0]
        future_predictions.append(prediction)
        for j in range(12, 1, -1):
            last_known_data[f'lag_{j}'] = last_known_data[f'lag_{j-1}']
        last_known_data['lag_1'] = prediction

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(product_data['Fecha'], product_data['Ingresos de Productos'], label='Datos historicos', color='blue')
    plt.scatter(future_dates, future_predictions, color='red', label='Prediccion a 6 meses')
    plt.plot(future_dates, future_predictions, color='red', linestyle='--', alpha=0.5)
    plt.title(f'Prediccion {product_name_dict[product_id]}')
    plt.xlabel('Datos historicos')
    plt.ylabel('Ingresos de Productos')
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Generar predicciones y gráficos para Ingresos de Productos
product_name_dict = data.set_index('ID del Producto')['Nombre del Producto'].to_dict()
for product_id in unique_products:
    predict_and_plot_for_product_revenue(product_id, rf_revenue, data_aggregated_revenue, product_name_dict)
