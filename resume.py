import os
import tensorflow as tf
from keras.optimizers import Adam
from callbacks.EvaluateCOCOMetricsCallback import EvaluateCOCOMetricsCallback
from keras.callbacks import TensorBoard
from utils.loadFiles import load_yolo_dataset
from utils.prepare import prepare_ds
from utils.visualize import visualize_detections
tf.config.optimizer.set_jit(False)
tf.keras.backend.clear_session()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Ocultar mensajes de advertencia

PATH_IMAGES = "/mnt/data3/sponte/datasets/conGSSSP.large/images"#"/Users/s.a.p.a/Documents/Datasets/conGSSSP/images/" # "D:\\Datasets\\conGSSSP_v2\\images\\" 
PATH_ANNOT = "/mnt/data3/sponte/datasets/conGSSSP.large/labels" #"/Users/s.a.p.a/Documents/Datasets/conGSSSP/labels/" # "D:\\Datasets\\conGSSSP_v2\\labels\\" 
BATCH_SIZE = 4
SPLIT_RATIO = 0.2
EPOCH = 10
SAVE_PATH = '/home/sponte/Repositorios/SpectroscopicObservationDetector/models/model.keras'
LEARNING_RATE = 0.001
GLOBAL_CLIPNORM = 10.0

# Etiquetas de clase
class_ids = [
    "observation"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Modelo
model = tf.keras.models.load_model(
    '/home/sponte/Repositorios/SpectroscopicObservationDetector/models/model.keras',
    compile=False
)
# Optimizador 
optimizer = Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM
)
# Compilar
model.compile(
    optimizer=optimizer,
    classification_loss="binary_crossentropy",
    box_loss='ciou',
    jit_compile=False
)

# Cargar dataset
train_data, val_data = load_yolo_dataset(PATH_IMAGES, PATH_ANNOT, SPLIT_RATIO)

# Preparar datos de entrenamiento
train_ds = prepare_ds(train_data, (640,640), BATCH_SIZE)
val_ds = prepare_ds(val_data, (640,640), BATCH_SIZE)

### Tuplas para el entrenamiento ###
# Funcion
def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]
# Entrenamiento
train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
# Validacion
val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Callbacks
callbacks = [
    EvaluateCOCOMetricsCallback(val_ds, SAVE_PATH),
    TensorBoard(
        log_dir='tensorboard/logdir', 
        histogram_freq=1,   # Frecuencia (en epocas) para generar histogramas
        update_freq="epoch",
    )
]

# Reanudar entrenamiento
model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=callbacks,
    epochs=EPOCH
)

### Visualizar predicciones ###
# Entrenamiento
visualize_detections(
    model=model, 
    dataset=train_ds, 
    bounding_box_format="rel_xywh",
    rows=2, cols=2, class_mapping=class_mapping, 
    save_path="train/train_predictions.png",
    confidence_threshold=0.5
)
# Validacion
visualize_detections(
    model=model, 
    dataset=val_ds, 
    bounding_box_format="rel_xywh",
    rows=2, cols=2, class_mapping=class_mapping, 
    save_path="train/val_predictions.png",
    confidence_threshold=0.5
)