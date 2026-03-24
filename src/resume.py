'''
Codigo para reanudar el entrenamiento de un modelo, ya sea con otro conjunto de datos
o con el mismo.
'''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Ocultar mensajes de advertencia
import tensorflow as tf
from keras.optimizers import Adam
from callbacks.EvaluateCOCOMetricsCallback import EvaluateCOCOMetricsCallback
from keras.callbacks import TensorBoard
from utils.loadFiles import load_yolo_dataset
from utils.prepare import prepare_ds
from utils.visualize import print_dataset_info_dict, visualize_detections
tf.config.optimizer.set_jit(False)
tf.keras.backend.clear_session()
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = '/home/sponte/Repositorios/SpectroscopicObservationDetector/models/model-v0.0.7.mr.keras'
PATH_IMAGES = "/mnt/data3/sponte/datasets/observaciones-etiquetadas/images.jpg" #"/Users/s.a.p.a/Documents/Datasets/conGSSSP/images/" # "D:\\Datasets\\conGSSSP_v2\\images\\" 
PATH_ANNOT = "/mnt/data3/sponte/datasets/observaciones-etiquetadas/labels" #"/Users/s.a.p.a/Documents/Datasets/conGSSSP/labels/" # "D:\\Datasets\\conGSSSP_v2\\labels\\" 
ROTATE_ANGLE = 90
SPLIT_RATIO = 0.2
EPOCH = 10

# Etiquetas de clase
class_ids = ["observation"]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

### Modelo ###
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)
optimizer = Adam( # Optimizador 
    learning_rate=float(os.getenv("LEARNING_RATE")),
    global_clipnorm=float(os.getenv("GLOBAL_CLIPNORM"))
)
model.compile( # Compilar modelo
    optimizer=optimizer,
    classification_loss="binary_crossentropy",
    box_loss='ciou',
    jit_compile=False
)

### Dataset ###
train_data, val_data = load_yolo_dataset(
    PATH_IMAGES, 
    PATH_ANNOT, 
    SPLIT_RATIO, 
    int(os.getenv("RANDOM_SEED")))
train_ds = prepare_ds( # Preparar datos entrenamiento
    train_data, 
    (640,640), 
    int(os.getenv("BATCH_SIZE")), 
    rotate_angle=ROTATE_ANGLE)
val_ds = prepare_ds( # Preparar datos prueba
    val_data, 
    (640,640), 
    int(os.getenv("BATCH_SIZE")), 
    rotate_angle=ROTATE_ANGLE)
# Informacion de datos
print_dataset_info_dict(train_ds, "Train")
print_dataset_info_dict(val_ds, "Validation")

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
    EvaluateCOCOMetricsCallback(val_ds, os.getenv("SAVE_PATH")),
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
    rows=4, cols=4, class_mapping=class_mapping, 
    save_path=os.path.join(os.getenv("PLOT_PATH"), "train_predictions.png"),
    confidence_threshold=0.5
)
# Validacion
visualize_detections(
    model=model, 
    dataset=val_ds, 
    bounding_box_format="rel_xywh",
    rows=4, cols=4, class_mapping=class_mapping, 
    save_path=os.path.join(os.getenv("PLOT_PATH"), "val_predictions.png"),
    confidence_threshold=0.5
)