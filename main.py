import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Ocultar mensajes
from tqdm.auto import tqdm
from callbacks.EvaluateCOCOMetricsCallback import EvaluateCOCOMetricsCallback
from utils.loadFiles import load_dataset, txt_to_annotation


import tensorflow as tf
from keras.optimizers import Adam

from utils.parse import center_rel_xywh_to_rel_xywh
from utils.resize import resize_rel_center_xywh
from utils.visualize import visualize_dataset

import keras_cv
# from keras_cv import bounding_box
# from keras_cv import visualization


SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0
SAVE_PATH = 'models/model.keras'

class_ids = [
    "observation"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Ubicacion de las imagenes y anotaciones
path_images = "/Users/s.a.p.a/Documents/Datasets/conGSSSP/images/" # "D:\\Datasets\\conGSSSP\\images\\"
path_annot = "/Users/s.a.p.a/Documents/Datasets/conGSSSP/labels/" # "D:\\Datasets\\conGSSSP\\labels\\"

# Recuperar archivos TXT de etiquetas ordenados
txt_files = sorted(
    [
        os.path.join(path_annot, file_name)
        for file_name in os.listdir(path_annot)
        if file_name.endswith(".txt")
    ]
)

# Recuperar archivos JPG de imagenes ordenados
jpg_files = sorted(
    [
        os.path.join(path_images, file_name)
        for file_name in os.listdir(path_images)
        if file_name.endswith(".jpg")
    ]
)

# Etiquetas: Aislar informacion de clases y cajas delimitadoras 
image_paths = []
bbox = []
classes = []
for txt_file in tqdm(txt_files):
    # Nombre del archivo de imagen correspondiente
    image_name = os.path.splitext(os.path.basename(txt_file))[0] + ".jpg"
    image_path = os.path.join(path_images, image_name)
    # Clase y cajas delimitadoras
    boxes, class_ids = txt_to_annotation(txt_file)
    # Agregar en listas
    image_paths.append(image_path)
    bbox.append(boxes)
    classes.append(class_ids)

# Manejo de casos vacios
processed_boxes = []
for b in bbox:
    if len(b) == 0:
        processed_boxes.append(tf.zeros((0, 4), dtype=tf.float32))
    else:
        processed_boxes.append(tf.convert_to_tensor(b, dtype=tf.float32))


# Cajas, Clases y Caminos de imagenes mejor como tensores
bbox = tf.ragged.constant(processed_boxes)
print('___________')
print(bbox.shape)
classes = tf.ragged.constant(classes)
print(bbox.shape)
image_paths = tf.ragged.constant(image_paths)

# Dataset final
data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

# Dividir conjunto de validacion y de test
num_val = int(len(txt_files) * SPLIT_RATIO)
val_data = data.take(num_val)
train_data = data.skip(num_val)


# Preparar datos de entrenamiento
train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.map(resize_rel_center_xywh, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(center_rel_xywh_to_rel_xywh, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)

# Preparar datos de validacion
val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(BATCH_SIZE * 4)
val_ds = val_ds.map(resize_rel_center_xywh, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(center_rel_xywh_to_rel_xywh, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)



# Visualizar
visualize_dataset(
    train_ds, bounding_box_format="rel_xywh", value_range=(0, 255), 
    rows=2, cols=2, class_mapping=class_mapping, save_path="train_visualization.png"
)

# Visualizar
visualize_dataset(
    val_ds, bounding_box_format="rel_xywh", value_range=(0, 255), 
    rows=2, cols=2, class_mapping=class_mapping, save_path="val_visualization.png"
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
)
# Optimizador 
optimizer = Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM
)
# Compilar
yolo.compile(
    optimizer=optimizer,
    classification_loss="binary_crossentropy",
    box_loss='ciou'
)

# Entrenar
yolo.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[EvaluateCOCOMetricsCallback(val_ds, SAVE_PATH)],
    epochs=EPOCH
)