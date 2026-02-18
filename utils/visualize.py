'''
Docstring for utils.visualize
Funciones para la visualizacion de datos
'''
from keras_cv import visualization
import matplotlib.pyplot as plt

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