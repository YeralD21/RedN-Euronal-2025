# Red Neuronal para Predicción de Rendimiento de Sistemas

Este proyecto implementa una red neuronal para predecir el rendimiento de un sistema informático basado en diferentes métricas del sistema.

## Características

- Predicción del rendimiento del sistema basado en:
  - Uso de CPU
  - Uso de memoria
  - Número de procesos
  - Tasa de transferencia de red
- Visualización del entrenamiento
- Función para realizar predicciones personalizadas

## Requisitos

Para ejecutar este proyecto, necesitas tener instalado Python 3.8 o superior y las siguientes dependencias:

```bash
pip install -r requirements.txt
```

## Uso

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecuta el script principal:
```bash
python red_neuronal_sistemas.py
```

El script:
- Generará datos sintéticos para el entrenamiento
- Entrenará la red neuronal
- Mostrará gráficos del proceso de entrenamiento
- Realizará una predicción de ejemplo

## Estructura del Proyecto

- `red_neuronal_sistemas.py`: Script principal con la implementación de la red neuronal
- `requirements.txt`: Lista de dependencias del proyecto
- `README.md`: Este archivo con las instrucciones

## Notas

- Los datos utilizados son sintéticos para fines de demostración
- El modelo puede ser adaptado para usar datos reales de monitoreo de sistemas
- La arquitectura de la red neuronal puede ser modificada según las necesidades específicas 