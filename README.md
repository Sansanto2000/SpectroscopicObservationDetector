# SpectroscopicObservationDetector

SOD (Spectroscopic Observation Detector) es un modelo de detección basado en aprendizaje profundo para la detección de observaciones en imágenes espectroscópicas. Basado en la familia de arquitecturas YOLO. 

El repositorio contine el codigo necesario para entrenar y ejecutar el modelo de deteccion.

# Entorno virtual

Se recomienda usar uv para la administracion del entorno virtual.

# Entrenamiento

Con el siguiente comando se puede iniciar el entrenamiento del modelo.
```
uv python main.py
```

## Tensorboard

Las metricas de ejecucion se pueden ver con el siguiente comando.
```
uv run tensorboard --logdir="tensorboard/logdir"
```


