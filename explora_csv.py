import pandas as pd

# Cargar el archivo CSV
archivo = 'multi_cloud_service_dataset.csv'
data = pd.read_csv(archivo)

# Mostrar las primeras filas
data_head = data.head()
print('Primeras filas del archivo:')
print(data_head)

# Mostrar las columnas disponibles
print('\nColumnas disponibles:')
print(list(data.columns)) 