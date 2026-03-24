import tensorflow as tf

def center_rel_xywh_to_rel_xywh(sample):
    """
    Desde una muestra convierte bounding boxes de formato rel_center_xywh a rel_xywh.

    Args:
        sample: _description_

    Returns:
        Muestra con boxes en formato rel_xywh (x_min, y_min, w, h)
    """
    boxes = sample["bounding_boxes"]["boxes"]
    boxes = tf.cast(boxes, tf.float32)

    x_c = boxes[..., 0:1]
    y_c = boxes[..., 1:2]
    w   = boxes[..., 2:3]
    h   = boxes[..., 3:4]

    x_min = x_c - w / 2.0
    y_min = y_c - h / 2.0

    new_boxes = tf.concat([x_min, y_min, w, h], axis=-1)

    return {
        "images": sample["images"],
        "bounding_boxes": {
            "boxes": new_boxes,
            "classes": sample["bounding_boxes"]["classes"],
        },
    }

def to_ragged_predictions(y_pred:dict) -> dict:
    boxes = y_pred["boxes"]
    classes = y_pred["classes"]
    confidence = y_pred["confidence"]
    num_det = y_pred["num_detections"]

    ragged_boxes = tf.RaggedTensor.from_tensor(
        boxes,
        lengths=num_det
    )

    ragged_classes = tf.RaggedTensor.from_tensor(
        classes,
        lengths=num_det
    )

    ragged_conf = tf.RaggedTensor.from_tensor(
        confidence,
        lengths=num_det
    )

    return {
        "boxes": ragged_boxes,
        "classes": ragged_classes,
        "confidence": ragged_conf,
    }