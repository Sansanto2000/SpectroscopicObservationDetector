'''
Codigo para el entrenamiento de modelos de deteccion de observaciones.
Deteccion basada en formato YOLO con kerasCV.
'''
import os
from utils.prepare import prepare_ds
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Ocultar mensajes de advertencia
from utils.loadFiles import load_yolo_dataset
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from callbacks.EvaluateCOCOMetricsCallback import EvaluateCOCOMetricsCallback
from utils.visualize import visualize_dataset, visualize_detections
import keras_cv
tf.config.optimizer.set_jit(False)
tf.keras.backend.clear_session()
from dotenv import load_dotenv

load_dotenv()
# Ubicacion de las imagenes y anotaciones
# PATH_IMAGES = "/mnt/data3/sponte/datasets/conGSSSP.large.3/images"
# PATH_ANNOT = "/mnt/data3/sponte/datasets/conGSSSP.large.3/labels" 
PATH_IMAGES = "/mnt/data3/sponte/datasets/observaciones-etiquetadas/images.jpg"
PATH_ANNOT = "/mnt/data3/sponte/datasets/observaciones-etiquetadas/labels"
ROTATE_ANGLE = 90
SPLIT_RATIO = 0.2
EPOCH = 100

# Etiquetas de clase
class_ids = [
    "observation"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Cargar dataset
train_data, val_data = load_yolo_dataset(
    PATH_IMAGES, 
    PATH_ANNOT, 
    SPLIT_RATIO, 
    int(os.getenv("RANDOM_SEED")))

### REGLAS DE LOS DATOS ###
"""
IMAGENES: (640, 640)
LABELS: rel_xywh
"""
# Preparar datos de entrenamiento
train_ds = prepare_ds(train_data, (640,640), int(os.getenv("BATCH_SIZE")), rotate_angle=ROTATE_ANGLE)
# Preparar datos de validacion
val_ds = prepare_ds(val_data, (640,640), int(os.getenv("BATCH_SIZE")), rotate_angle=ROTATE_ANGLE)

# Visualizar
visualize_dataset(
    train_ds, bounding_box_format="rel_xywh", value_range=(0, 255), 
    rows=4, cols=4, class_mapping=class_mapping, 
    save_path=os.path.join(os.getenv("PLOT_PATH"), "train_visualization.png")
)

# Visualizar
visualize_dataset(
    val_ds, bounding_box_format="rel_xywh", value_range=(0, 255), 
    rows=4, cols=4, class_mapping=class_mapping, 
    save_path=os.path.join(os.getenv("PLOT_PATH"), "val_visualization.png")
)

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

### Modelo ###
# Esqueleto YOLO
backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone_coco"  # small backbone con pesos de COCO
)
# Modelo
yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="rel_xywh",
    backbone=backbone,
    fpn_depth=1,
    # prediction_decoder=keras_cv.layers.MultiClassNonMaxSuppression(
    #     bounding_box_format="rel_xywh",
    #     from_logits=False,
    #     confidence_threshold=0.5,
    #     iou_threshold=0.5,
    # ),
)
# Optimizador 
optimizer = Adam(
    learning_rate=float(os.getenv("LEARNING_RATE")),
    global_clipnorm=float(os.getenv("GLOBAL_CLIPNORM"))
)
# Compilar
yolo.compile(
    optimizer=optimizer,
    classification_loss="binary_crossentropy",
    box_loss='ciou',
    jit_compile=False
)

callbacks = [
    EvaluateCOCOMetricsCallback(val_ds, os.getenv("SAVE_PATH")),
    TensorBoard(
        log_dir='tensorboard/logdir', 
        histogram_freq=1,   # Frecuencia (en epocas) para generar histogramas
        update_freq="epoch",
    )
]

# Entrenar
yolo.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=callbacks,
    epochs=EPOCH
)


### Visualizar predicciones ###
# Entrenamiento
visualize_detections(
    yolo, 
    dataset=train_ds, 
    bounding_box_format="rel_xywh",
    rows=4, cols=4, class_mapping=class_mapping, 
    save_path=os.path.join(os.getenv("PLOT_PATH"), "train_predictions.png"),
    confidence_threshold=0.5
)
# Validacion
visualize_detections(
    yolo, 
    dataset=val_ds, 
    bounding_box_format="rel_xywh",
    rows=4, cols=4, class_mapping=class_mapping, 
    save_path=os.path.join(os.getenv("PLOT_PATH"), "val_predictions.png"),
    confidence_threshold=0.5
)