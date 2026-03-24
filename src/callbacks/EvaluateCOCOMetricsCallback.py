from keras.callbacks import Callback
import keras_cv
import tensorflow as tf

from utils.parse import to_ragged_predictions

class EvaluateCOCOMetricsCallback(Callback):
    """
    Docstring para EvaluateCOCOMetricsCallback
    Al final de cada epoca se evaluan las metricas MaP e IoU,
    relacionadas al conjunto de datos COCO. Si el valor de MaP
    mejora se guarda el modelo en la ruta especificada.
    """
    
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xywh",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for images, y_true in self.data:
            y_pred = self.model.predict(images, verbose=0)
            y_pred = to_ragged_predictions(y_pred)

            # Convertir piso de verdad a absoluto
            y_true = keras_cv.bounding_box.convert_format(
                y_true,
                source="rel_xywh",
                target="xywh",
                images=images,
            )

            # Convertir predicciones a absoluto
            y_pred = keras_cv.bounding_box.convert_format(
                y_pred,
                source="rel_xywh",
                target="xywh",
                images=images,
            )
            #y_pred =  tf.RaggedTensor.from_tensor() tf.ragged.constant(y_pred)
            
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        # Si MaP mejora guarda el modelo
        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)  

        return logs