# SpectroscopicObservationDetector

SOD (Spectroscopic Observation Detector) es un modelo de detección basado en aprendizaje profundo para la detección de observaciones en imágenes espectroscópicas. Basado en la familia de arquitecturas YOLO. 

El repositorio contine el codigo necesario para entrenar y ejecutar el modelo de deteccion.

# Entorno virtual

Se recomienda usar uv para la administracion del entorno virtual.

# Entrenamiento

Con el siguiente comando se puede iniciar el entrenamiento del modelo.
```
uv run src/main.py
# o
nohup uv run src/main.py > output.log 2>&1 &
```

Con el siguiente comando se puede reanudar el entrenamiento de un modelo que ya estubiese entrenando.
```
uv run src/resume.py
```

# Evaluacion

Con el siguiete comando se puede evaluar las metricas *IoU* y *Map* del modelo entrenado.
```
uv run src/evaluate.py
```

## Tensorboard

Las metricas de ejecucion se pueden ver con el siguiente comando.
```
uv run tensorboard --logdir="tensorboard/logdir"
```


