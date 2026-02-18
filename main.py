import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Ocultar mensajes
from tqdm.auto import tqdm
from utils.loadFiles import load_dataset, txt_to_annotation


import tensorflow as tf

from utils.parse import center_rel_xywh_to_rel_xywh
from utils.resize import resize_rel_center_xywh
from utils.visualize import visualize_dataset
# from keras_cv import bounding_box
# from keras_cv import visualization

SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5

class_ids = [
    "observation"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Ubicacion de las imagenes y anotaciones
path_images = "D:\\Datasets\\conGSSSP\\images\\"
path_annot = "D:\\Datasets\\conGSSSP\\labels\\"

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

# Cajas, Clases y Caminos de imagenes mejor como tensores
bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
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