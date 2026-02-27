'''
Docstring for utils.txtToAnnotation
Funciones para la carga de archivos en formatos compatibles.
'''
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from tqdm import tqdm

def load_yolo_dataset(path_images:str, path_annot:str, SPLIT_RATIO:float) -> tuple[tf.RaggedTensor, tf.RaggedTensor]:
    """Cargar dataset de imagenes anotadas en base a archivos TXT de anotaciones y archivos JPG de imagenes.

    Args:
        path_images (str): path del directorio de imagenes JPG.
        path_annot (str): path del directorio de anotaciones TXT.
        SPLIT_RATIO (float): proporcion de datos para validacion.

    Returns:
        tuple[tf.RaggedTensor, tf.RaggedTensor]: dataset de entrenamiento y dataset de validacion.
    """


    print("Cargando dataset...")
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
        try:
            boxes, class_ids = txt_to_annotation(txt_file)
        except Exception as e:
            print(f"Error en archivo: {txt_file}")
            raise e
        # Agregar en listas
        image_paths.append(image_path)
        bbox.append(boxes)
        classes.append(class_ids)


    # Cajas, Clases y Caminos de imagenes mejor como tensores
    bbox = tf.ragged.constant(bbox, ragged_rank=1, inner_shape=(4,))
    classes = tf.ragged.constant(classes)
    image_paths = tf.ragged.constant(image_paths)

    # Dataset final
    data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

    # Dividir conjunto de validacion y de test
    num_val = int(len(txt_files) * SPLIT_RATIO)
    val_data = data.take(num_val)
    train_data = data.skip(num_val)

    return train_data, val_data

def txt_to_annotation(txt_file):
    """_summary_
    Leer archivo TXT indicado y retornar la informacion de su caja delimitadora.

    Args:
        txt_file:string - Path del archivo TXT a leer.

    Return:
        boxes: lista de cajas delimitadoras del archivo.
        class_ids: lista de clases de cada caja delimitadora.
    """
    boxes = []
    class_ids = []

    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        # Saltear lineas vacias
        if len(parts) == 0:
            continue
        # Separar valores
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        boxes.append([x_center, y_center, width, height])
        class_ids.append(class_id)

    return boxes, class_ids


def load_image(image_path):
    """Cargar imagen como tensor.

    Args:
    - image_path:string - Path de la imagen.

    Return:
    - image - Tensor con informacion de imagen shape=(..., 3)
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    return image


def load_dataset(image_path, classes, bbox):
    """_summary_
    Prepara diccionario con informacion de imagen, sus clases y sus 
    cajas delimitadoras.

    Args:
    - image_path:string - Path de la imagen.
    - classes:int[] - Etiquetas numericas que representan las 
    clases de cada caja delimitadora.
    - bbox:float[][] - Arreglo de arreglos de 4 elementos flotantes
    entre 0 y 1 que representan las coordenadas de cada caja 
    delimitadora de la imagen. 

    Return:
    - _ - Diccionario con informacion de imagen y sus cajas delimitadoras.
    """
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": tf.cast(bbox, tf.float32),
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}