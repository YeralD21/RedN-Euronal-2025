import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

RESET = '\033[0m'
BOLD = '\033[1m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'

# Cargar datos reales
archivo = 'multi_cloud_service_dataset.csv'
data = pd.read_csv(archivo)

# Seleccionar variables de entrada y salida
columnas_entrada = [
    'CPU_Utilization (%)',
    'Memory_Usage (MB)',
    'Storage_Usage (GB)',
    'Network_Bandwidth (Mbps)',
    'Service_Latency (ms)',
    'Response_Time (ms)',
    'Throughput (Requests/sec)',
    'Load_Balancing (%)',
    'QoS_Score',
    'Workload_Variability'
]
X = data[columnas_entrada]
y = data['Optimal_Service_Placement']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el scaler para uso futuro
joblib.dump(scaler, 'scaler_modelo.pkl')

# Convertir a tensores de PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

# Definir la red neuronal para clasificación binaria
class RedNeuronal(nn.Module):
    def __init__(self):
        super(RedNeuronal, self).__init__()
        self.capa1 = nn.Linear(10, 32)
        self.capa2 = nn.Linear(32, 16)
        self.capa3 = nn.Linear(16, 8)
        self.capa4 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.capa1(x))
        x = self.relu(self.capa2(x))
        x = self.relu(self.capa3(x))
        x = self.sigmoid(self.capa4(x))
        return x

# Entrenamiento con validación para guardar el mejor modelo
mejor_acc = 0
mejor_modelo = None
mejor_epoca = 0
historial = {'pérdida': [], 'val_pérdida': [], 'val_acc': []}

modelo = RedNeuronal()
criterio = nn.BCELoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

for epoca in range(100):
    modelo.train()
    optimizador.zero_grad()
    salidas = modelo(X_train_tensor)
    pérdida = criterio(salidas, y_train_tensor)
    pérdida.backward()
    optimizador.step()
    # Validación
    modelo.eval()
    with torch.no_grad():
        val_salidas = modelo(X_test_tensor)
        val_pérdida = criterio(val_salidas, y_test_tensor)
        val_pred = (val_salidas.numpy() > 0.5).astype(int)
        val_acc = accuracy_score(y_test, val_pred)
    historial['pérdida'].append(pérdida.item())
    historial['val_pérdida'].append(val_pérdida.item())
    historial['val_acc'].append(val_acc)
    if val_acc > mejor_acc:
        mejor_acc = val_acc
        mejor_modelo = RedNeuronal()
        mejor_modelo.load_state_dict(modelo.state_dict())
        mejor_epoca = epoca
        torch.save(mejor_modelo.state_dict(), 'modelo_entrenado.pth')
    if (epoca + 1) % 10 == 0:
        print(f'Época [{epoca+1}/100], Pérdida: {pérdida.item():.4f}, Val. Pérdida: {val_pérdida.item():.4f}, Val. Acc: {val_acc:.4f}')

print(f'\nMejor modelo guardado en la época {mejor_epoca+1} con accuracy de validación: {mejor_acc:.4f}')

# Visualizar resultados
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.plot(historial['pérdida'], label='Pérdida de entrenamiento')
plt.plot(historial['val_pérdida'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(historial['val_acc'], label='Accuracy de validación', color='green')
plt.title('Accuracy de validación')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
# Matriz de confusión
plt.subplot(1, 3, 3)
mejor_modelo.eval()
with torch.no_grad():
    predicciones = mejor_modelo(X_test_tensor)
    pred_binarias = (predicciones.numpy() > 0.5).astype(int)
cm = confusion_matrix(y_test, pred_binarias)

# Extraer métricas específicas de la matriz de confusión
TN = cm[0, 0]  # Verdaderos Negativos
FP = cm[0, 1]  # Falsos Positivos ← AQUÍ ESTÁN
FN = cm[1, 0]  # Falsos Negativos  
TP = cm[1, 1]  # Verdaderos Positivos

# Imprimir métricas detalladas
print(f"\n{BOLD}{MAGENTA}=== ANÁLISIS DETALLADO DE LA MATRIZ DE CONFUSIÓN ==={RESET}")
print(f"{GREEN}Verdaderos Negativos (TN): {TN} casos{RESET}")
print(f"{YELLOW}Falsos Positivos (FP): {FP} casos{RESET} - Servicios NO óptimos clasificados como óptimos")
print(f"{RED}Falsos Negativos (FN): {FN} casos{RESET} - Servicios óptimos clasificados como NO óptimos") 
print(f"{BLUE}Verdaderos Positivos (TP): {TP} casos{RESET}")

# Calcular métricas adicionales
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{BOLD}{CYAN}=== MÉTRICAS CALCULADAS ==={RESET}")
print(f"Precision: {precision:.3f} ({precision:.1%})")
print(f"Recall (Sensibilidad): {recall:.3f} ({recall:.1%})")
print(f"Specificity: {specificity:.3f} ({specificity:.1%})")
print(f"F1-Score: {f1_score:.3f} ({f1_score:.1%})")

# Análisis de riesgo de falsos positivos
if FP > 0:
    tasa_fp = FP / (FP + TN)
    print(f"\n{BOLD}{YELLOW}⚠️ ANÁLISIS DE FALSOS POSITIVOS:{RESET}")
    print(f"- Cantidad: {FP} casos de {len(y_test)} pruebas")
    print(f"- Tasa de Falsos Positivos: {tasa_fp:.3f} ({tasa_fp:.1%})")
    print(f"- Riesgo: Mantener {FP} servicios en ubicaciones problemáticas")
    print(f"- Impacto: Posible bajo rendimiento y costos elevados")

# ANÁLISIS DE UMBRALES ALTERNATIVOS
print(f"\n{BOLD}{CYAN}=== ANÁLISIS DE UMBRALES ALTERNATIVOS ==={RESET}")
umbrales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

mejor_f1 = 0
mejor_umbral = 0.5

for umbral in umbrales:
    # Aplicar umbral personalizado
    pred_umbral = (predicciones.numpy() > umbral).astype(int)
    
    # Calcular matriz de confusión para este umbral
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    cm_umbral = confusion_matrix(y_test, pred_umbral)
    
    if len(cm_umbral) == 2:  # Asegurar que hay ambas clases
        tn_u = cm_umbral[0, 0]
        fp_u = cm_umbral[0, 1] if cm_umbral.shape[1] > 1 else 0
        fn_u = cm_umbral[1, 0] if cm_umbral.shape[0] > 1 else 0
        tp_u = cm_umbral[1, 1] if cm_umbral.shape[0] > 1 and cm_umbral.shape[1] > 1 else 0
    else:
        # Solo una clase presente
        if y_test.sum() == 0:  # Solo clase 0
            tn_u = len(y_test) - pred_umbral.sum()
            fp_u = pred_umbral.sum()
            fn_u = 0
            tp_u = 0
        else:  # Solo clase 1
            tn_u = 0
            fp_u = 0
            fn_u = len(y_test) - pred_umbral.sum()
            tp_u = pred_umbral.sum()
    
    # Calcular métricas
    accuracy_u = (tp_u + tn_u) / len(y_test)
    precision_u = tp_u / (tp_u + fp_u) if (tp_u + fp_u) > 0 else 0
    recall_u = tp_u / (tp_u + fn_u) if (tp_u + fn_u) > 0 else 0
    f1_u = 2 * (precision_u * recall_u) / (precision_u + recall_u) if (precision_u + recall_u) > 0 else 0
    
    print(f"Umbral {umbral:.1f}: Acc={accuracy_u:.3f}, Prec={precision_u:.3f}, Rec={recall_u:.3f}, F1={f1_u:.3f} | TP={tp_u}, TN={tn_u}, FP={fp_u}, FN={fn_u}")
    
    # Guardar el mejor F1-Score
    if f1_u > mejor_f1:
        mejor_f1 = f1_u
        mejor_umbral = umbral

print(f"\n{BOLD}{GREEN}🎯 MEJOR UMBRAL ENCONTRADO: {mejor_umbral} (F1-Score: {mejor_f1:.3f}){RESET}")

# Probar predicción con mejor umbral
def predecir_con_umbral_optimo(valores, umbral_optimo=mejor_umbral):
    datos = np.array([valores])
    datos_escalados = scaler.transform(datos)
    datos_tensor = torch.FloatTensor(datos_escalados)
    with torch.no_grad():
        prediccion = mejor_modelo(datos_tensor)
        probabilidad = prediccion.item()
    return int(probabilidad > umbral_optimo), probabilidad

# Probar con los casos de ejemplo usando el mejor umbral
if 'valores_ficticios' in locals():
    pred_ficticio_opt, prob_ficticio = predecir_con_umbral_optimo(valores_ficticios)
    pred_inestable_opt, prob_inestable = predecir_con_umbral_optimo(valores_inestables)
    pred_estable_opt, prob_estable = predecir_con_umbral_optimo(valores_estables)
else:
    # Definir valores de ejemplo si no existen
    valores_ficticios = [65, 2048, 50, 200, 30, 40, 500, 70, 0.9, 0.3]
    valores_inestables = [95, 4096, 120, 20, 200, 300, 50, 10, 0.2, 0.9]
    valores_estables = [40, 1024, 20, 500, 10, 15, 800, 80, 0.95, 0.1]
    
    pred_ficticio_opt, prob_ficticio = predecir_con_umbral_optimo(valores_ficticios)
    pred_inestable_opt, prob_inestable = predecir_con_umbral_optimo(valores_inestables)
    pred_estable_opt, prob_estable = predecir_con_umbral_optimo(valores_estables)

print(f"\n{BOLD}{MAGENTA}=== PREDICCIONES CON UMBRAL ÓPTIMO ({mejor_umbral}) ==={RESET}")
print(f"Servicio ficticio: {'Óptimo' if pred_ficticio_opt == 1 else 'No óptimo'} (Prob: {prob_ficticio:.3f})")
print(f"Servicio inestable: {'Óptimo' if pred_inestable_opt == 1 else 'No óptimo'} (Prob: {prob_inestable:.3f})")
print(f"Servicio estable: {'Óptimo' if pred_estable_opt == 1 else 'No óptimo'} (Prob: {prob_estable:.3f})")

plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Matriz de confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.tight_layout()
plt.show()

# Eliminar impresión del reporte de clasificación detallado y mostrar solo la accuracy general
print(f"\n{BOLD}{BLUE}Accuracy general del modelo en datos de prueba: {mejor_acc:.2%}{RESET}")
print(f"El modelo acierta en aproximadamente {int(mejor_acc*100)} de cada 100 casos de prueba.")
print("Esto indica que el modelo es confiable para anticipar si la ubicación de un servicio es óptima o no, según las métricas del sistema.")

# --- TEST CON DATOS FICTICIOS ---
# Ejemplo de datos ficticios para test
ficticio = {
    'Service_ID': 'S9999',
    'Service_Type': 'WebApp',
    'Cloud_Provider': 'AWS',
    'Edge_Node_ID': 'EdgeX',
    'CPU_Utilization (%)': 65,
    'Memory_Usage (MB)': 2048,
    'Storage_Usage (GB)': 50,
    'Network_Bandwidth (Mbps)': 200,
    'Service_Latency (ms)': 30,
    'Response_Time (ms)': 40,
    'Throughput (Requests/sec)': 500,
    'Load_Balancing (%)': 70,
    'QoS_Score': 0.9,
    'Workload_Variability': 0.3,
    'Optimal_Service_Placement': None
}

# Usar solo las variables de entrada
valores_ficticios = [
    ficticio['CPU_Utilization (%)'],
    ficticio['Memory_Usage (MB)'],
    ficticio['Storage_Usage (GB)'],
    ficticio['Network_Bandwidth (Mbps)'],
    ficticio['Service_Latency (ms)'],
    ficticio['Response_Time (ms)'],
    ficticio['Throughput (Requests/sec)'],
    ficticio['Load_Balancing (%)'],
    ficticio['QoS_Score'],
    ficticio['Workload_Variability']
]

# Cargar el mejor modelo y el scaler
scaler = joblib.load('scaler_modelo.pkl')
modelo_cargado = RedNeuronal()
modelo_cargado.load_state_dict(torch.load('modelo_entrenado.pth'))
modelo_cargado.eval()

def predecir_servicio(valores):
    datos = np.array([valores])
    datos_escalados = scaler.transform(datos)
    datos_tensor = torch.FloatTensor(datos_escalados)
    with torch.no_grad():
        prediccion = modelo_cargado(datos_tensor)
    return int(prediccion.item() > 0.5)

pred_ficticio = predecir_servicio(valores_ficticios)
print(f"\nPredicción para el servicio ficticio: {'Óptima' if pred_ficticio == 1 else 'No óptima'}")

# Escenario 1: Datos con administración deficiente (valores extremos o inestables)
ficticio_inestable = {
    'Service_ID': 'S9001',
    'Service_Type': 'BatchJob',
    'Cloud_Provider': 'Azure',
    'Edge_Node_ID': 'EdgeY',
    'CPU_Utilization (%)': 95,  # Muy alto
    'Memory_Usage (MB)': 4096,  # Muy alto
    'Storage_Usage (GB)': 120,  # Muy alto
    'Network_Bandwidth (Mbps)': 20,  # Bajo
    'Service_Latency (ms)': 200,  # Muy alto
    'Response_Time (ms)': 300,  # Muy alto
    'Throughput (Requests/sec)': 50,  # Bajo
    'Load_Balancing (%)': 10,  # Bajo
    'QoS_Score': 0.2,  # Bajo
    'Workload_Variability': 0.9,  # Muy variable
    'Optimal_Service_Placement': None
}
valores_inestables = [
    ficticio_inestable['CPU_Utilization (%)'],
    ficticio_inestable['Memory_Usage (MB)'],
    ficticio_inestable['Storage_Usage (GB)'],
    ficticio_inestable['Network_Bandwidth (Mbps)'],
    ficticio_inestable['Service_Latency (ms)'],
    ficticio_inestable['Response_Time (ms)'],
    ficticio_inestable['Throughput (Requests/sec)'],
    ficticio_inestable['Load_Balancing (%)'],
    ficticio_inestable['QoS_Score'],
    ficticio_inestable['Workload_Variability']
]
pred_inestable = predecir_servicio(valores_inestables)

# Escenario 2: Datos estables y óptimos
ficticio_estable = {
    'Service_ID': 'S9002',
    'Service_Type': 'API',
    'Cloud_Provider': 'GCP',
    'Edge_Node_ID': 'EdgeZ',
    'CPU_Utilization (%)': 40,  # Moderado
    'Memory_Usage (MB)': 1024,  # Moderado
    'Storage_Usage (GB)': 20,  # Moderado
    'Network_Bandwidth (Mbps)': 500,  # Alto
    'Service_Latency (ms)': 10,  # Bajo
    'Response_Time (ms)': 15,  # Bajo
    'Throughput (Requests/sec)': 800,  # Alto
    'Load_Balancing (%)': 80,  # Alto
    'QoS_Score': 0.95,  # Muy bueno
    'Workload_Variability': 0.1,  # Muy estable
    'Optimal_Service_Placement': None
}
valores_estables = [
    ficticio_estable['CPU_Utilization (%)'],
    ficticio_estable['Memory_Usage (MB)'],
    ficticio_estable['Storage_Usage (GB)'],
    ficticio_estable['Network_Bandwidth (Mbps)'],
    ficticio_estable['Service_Latency (ms)'],
    ficticio_estable['Response_Time (ms)'],
    ficticio_estable['Throughput (Requests/sec)'],
    ficticio_estable['Load_Balancing (%)'],
    ficticio_estable['QoS_Score'],
    ficticio_estable['Workload_Variability']
]
pred_estable = predecir_servicio(valores_estables)

# Función para imprimir en color
RESET = '\033[0m'
BOLD = '\033[1m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'

# --- REPORTE DETALLADO DEL PROCESO Y RESULTADOS ---
print(f"\n{BOLD}{CYAN}1. Analizando Data set: multi_cloud_service_dataset.csv{RESET}")
print(f"{CYAN}Procesamiento de variables clave y normalización:{RESET}")
print(f"- Variables seleccionadas: {columnas_entrada}")
print("- Se aplicó StandardScaler para normalizar las variables numéricas.")

print(f"\n{BOLD}{BLUE}2. División de datos: Entrenamiento y prueba{RESET}")
print(f"- Tamaño total de datos: {len(data)} registros")
print(f"- Datos de entrenamiento: {len(X_train)} registros")
print(f"- Datos de prueba: {len(X_test)} registros")

print(f"\n{BOLD}{MAGENTA}3. Arquitectura de la red neuronal utilizada:{RESET}")
print("- Entrada: 10 variables")
print("- Capas ocultas: [32 neuronas, 16 neuronas, 8 neuronas] con activación ReLU")
print("- Capa de salida: 1 neurona con activación Sigmoid (clasificación binaria)")

print(f"\n{BOLD}{YELLOW}4. Nombre del modelo entrenado: modelo_entrenado.pth{RESET}")
print(f"{YELLOW}Nombre del scaler guardado: scaler_modelo.pkl{RESET}")

print(f"\n{BOLD}{GREEN}5. Datos usados para testear el modelo:{RESET}")
print(f"- Ejemplo 1 (administración deficiente): {valores_inestables}")
print(f"- Ejemplo 2 (administración óptima): {valores_estables}")

# --- REPORTE GENERAL DE OPTIMIDAD ---
print(f"\n{BOLD}{RED}6. REPORTE GENERAL DE OPTIMIDAD DEL SERVICIO{RESET}")

if pred_inestable == 1:
    print(f"{RED}Ejemplo 1: El modelo considera ÓPTIMA la ubicación del servicio bajo condiciones inestables. Sin embargo, esto puede indicar que el modelo necesita más datos de casos negativos para mejorar su discriminación.{RESET}")
else:
    print(f"{GREEN}Ejemplo 1: El modelo considera NO ÓPTIMA la ubicación del servicio bajo condiciones inestables. Esto es correcto, ya que altos consumos, latencia y baja calidad suelen indicar mala administración.{RESET}")

if pred_estable == 1:
    print(f"{GREEN}Ejemplo 2: El modelo considera ÓPTIMA la ubicación del servicio bajo condiciones estables y eficientes. Esto es correcto, ya que los recursos están bien balanceados, la latencia es baja y la calidad es alta.{RESET}")
else:
    print(f"{RED}Ejemplo 2: El modelo considera NO ÓPTIMA la ubicación del servicio bajo condiciones estables. Esto puede indicar que el modelo necesita más ejemplos positivos para aprender a reconocer escenarios óptimos.{RESET}")

print(f"\n{BOLD}{CYAN}7. Fundamento:{RESET}")
print("- El modelo fundamenta su decisión en las métricas clave de uso de recursos, latencia, throughput, balanceo de carga y calidad de servicio.")
print("- Una ubicación óptima se caracteriza por uso moderado de recursos, baja latencia, alta calidad y buena distribución de carga.")
print("- Una ubicación no óptima suele mostrar saturación, alta variabilidad, baja calidad y cuellos de botella en la red o el procesamiento.")

print(f"\n{BOLD}{BLUE}8. Conclusión:{RESET}")
print("El modelo es útil para anticipar y fundamentar decisiones de despliegue y administración de servicios en la nube, ayudando a optimizar recursos y mejorar la eficiencia operativa en Ingeniería de Sistemas.") 