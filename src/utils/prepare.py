import keras_cv
from utils.loadFiles import load_dataset
from utils.parse import center_rel_xywh_to_rel_xywh
from utils.resize import resize_rel_center_xywh
import tensorflow as tf


def prepare_ds(ds, target_size, batch_size, rotate_angle=None):
    """Recibe un conjunto de datos YOLO sin imagenes cargadas con 
    etiquetas en formato xywh y las prepara para el entrenamiento.
    - Carga las imagenes como tensores.
    - Divide los datos en lotes.
    - Convierte las etiquetas a formato rel_xywh.
    - Redimensiona la imagen a tamaño {{target}}

    Args:
        ds (_type_): dataset
        target_size (_type_): tamaño de redimension objetivo.
        batch_size (_type_): tamaño de lote.
        rotate_angle (_type_): angulo de rotacion de las imagenes.

    Returns:
        _type_: dataset preparado
    """

    ds = ds.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(batch_size * 4)
    ds = ds.map(
        lambda x: resize_rel_center_xywh(x, target_size), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(center_rel_xywh_to_rel_xywh, num_parallel_calls=tf.data.AUTOTUNE)

    if rotate_angle is not None:
        factor = rotate_angle / 360.0
        augmenter = keras_cv.layers.RandomRotation(
            factor=(factor, factor),  # 90° exactos (0.25 = 90°)
            bounding_box_format="rel_xywh"
        )
        ds = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.ragged_batch(batch_size, drop_remainder=True)
    return ds