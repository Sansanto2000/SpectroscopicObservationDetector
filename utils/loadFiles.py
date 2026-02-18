'''
Docstring for utils.txtToAnnotation
Funciones para la carga de archivos en formatos compatibles.
'''
from matplotlib import pyplot as plt
import tensorflow as tf


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
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        boxes.append([x_center, y_center, width, height])
        class_ids.append(class_id)

    return boxes, class_ids


def load_image(image_path):
    """_summary_
    Cargar imagen como tensor.

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