import tensorflow as tf

def resize_rel_center_xywh(sample):
    """_summary_
    Redimensiona la imagen de la muestra recibida con las proporciones 
    indicadas. Mantiene la relacion de aspecto de las cajas delimitadoras. 
    Formato: rel_xywh.

    Args:
        sample (_type_): _description_
        shape (_type_): _description_

    Returns:
        _type_: entrada redimencionada
    """
    target=(640,640)
    image = sample["images"]
    boxes = sample["bounding_boxes"]["boxes"]

    # Alto y ancho base
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    # Escala del lado mas reducido
    scale = tf.minimum(
        target[0] / tf.cast(h, tf.float32),
        target[1] / tf.cast(w, tf.float32),
    )
    # Nueva altura, nuevo ancho
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    # Cantidad de relleno necesario en alto y ancho
    pad_y = (target[0] - new_h) // 2
    pad_x = (target[1] - new_w) // 2

    # Redimensionar imagen
    image = tf.image.resize_with_pad(image, target[0], target[1])

    ### Ajustar cajas (si están en rel_center_xywh) ###
    # Datos
    x  = boxes[..., 0:1]
    y  = boxes[..., 1:2]
    bw = boxes[..., 2:3]
    bh = boxes[..., 3:4]

    # Nuevo x e y en pixeles
    x = x * tf.cast(new_w, tf.float32) + tf.cast(pad_x, tf.float32)
    y = y * tf.cast(new_h, tf.float32) + tf.cast(pad_y, tf.float32)
    # Coordenadas relativas
    x = x / target[1]
    y = y / target[0]
    bw = (bw * w * scale) / target[0]
    bh = (bh * h * scale) / target[1]
    # Nuevas coordenadas
    new_boxes = tf.concat([x, y, bw, bh], axis=-1)

    return {
        "images": image,
        "bounding_boxes": {
            "boxes": new_boxes,
            "classes": sample["bounding_boxes"]["classes"],
        },
    }