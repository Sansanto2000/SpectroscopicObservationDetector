'''
Docstring for utils.visualize
Funciones para la visualizacion de datos
'''
from keras_cv import visualization
import matplotlib.pyplot as plt
from keras_cv import bounding_box
import tensorflow as tf

def visualize_dataset(
        inputs, 
        value_range, 
        rows, 
        cols, 
        bounding_box_format, 
        class_mapping, 
        save_path=None
    ):
    """_summary_

    Args:
        inputs (_type_): _description_
        value_range (_type_): _description_
        rows (_type_): _description_
        cols (_type_): _description_
        bounding_box_format (_type_): _description_
        class_mapping (_type_): _description_
    """
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=1,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

def visualize_detections(model, dataset, bounding_box_format,
                         rows, cols, class_mapping, save_path=None,
                         confidence_threshold=None):
    """Visualiza las detecciones realizadas por un modelo.

    Args:
        model (_type_): Modelo que realiza las predicciones.
        dataset (_type_): Conjunto de datos
        bounding_box_format (_type_): Formato de las cajas 
            delimitadoras.
        rows (_type_): _description_
        cols (_type_): _description_
        class_mapping (_type_): _description_
        save_path (_type_, optional): _description_. Defaults to None.
    """
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    if(confidence_threshold != None):
        y_pred = filter_predictions(y_pred, confidence_threshold)

    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=rows,
        cols=cols,
        show=False,
        font_scale=0.7,
        class_mapping=class_mapping,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def filter_predictions(y_pred, threshold=0.5):
    mask = y_pred["confidence"] >= threshold

    filtered = {
        "boxes": tf.ragged.boolean_mask(y_pred["boxes"], mask),
        "classes": tf.ragged.boolean_mask(y_pred["classes"], mask),
        "confidence": tf.ragged.boolean_mask(y_pred["confidence"], mask),
    }
    return filtered