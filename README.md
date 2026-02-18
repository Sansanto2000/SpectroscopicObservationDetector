# SpectroscopicObservationDetector

SOD (Spectroscopic Observation Detector) es un modelo de detección basado en aprendizaje profundo para la detección de observaciones en imágenes espectroscópicas. Basado en la familia de arquitecturas YOLO. 

El repositorio contine el codigo necesario para entrenar y ejecutar el modelo de deteccion.

# Entorno virtual

Se recomienda usar un entorno virtual para manejar las dependencias del codigo ().

🔨 Crear entorno virtual `.\venv`:
```
python -m venv venv
``` 

🚀 Activar entorno virtual `.\venv`:
```
# Windows
.\venv\Scripts\Activate.ps1

# Mac
source venv/bin/activate
```

# Dependencias

📦 Instala las dependencias necesarias con:
```
pip install -r requirements.txt
```

# Entrenamiento

Con el siguiente comando se puede iniciar el entrenamiento del modelo.
```
python main.py
```


