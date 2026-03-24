import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Ocultar mensajes de advertencia
import tensorflow as tf
from keras.optimizers import Adam
from callbacks.EvaluateCOCOMetricsCallback import EvaluateCOCOMetricsCallback
from keras.callbacks import TensorBoard
from utils.loadFiles import load_yolo_dataset
from utils.prepare import prepare_ds
from utils.visualize import visualize_detections
tf.config.optimizer.set_jit(False)
tf.keras.backend.clear_session()

SAVE_PATH = '/home/sponte/Repositorios/SpectroscopicObservationDetector/models/model.keras'
MODEL_INPUT_SIZE = 640

# Modelo
model = tf.keras.models.load_model(
    '/home/sponte/Repositorios/SpectroscopicObservationDetector/models/model.keras',
    compile=False
)

# Input
dummy = tf.random.normal([1, 640, 640, 3])

# Prediccion
result = model.predict(dummy)
print('---------------Ejemplo Output---------------')
print(type(result))
print(result)
print('--------------------------------------------')

### Exportable para JS ###
# Definir interfaz entrada\salida del modelo a exportar
@tf.function(input_signature=[tf.TensorSpec([1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3], tf.float32)])
def serving_fn(x):
    y = model(x, training=False)
    print('--------------------------------------------')
    print("TYPE:", type(y))
    print("Y:", y)
    print('--------------------------------------------')
    return {
        "boxes": tf.cast(y["boxes"], tf.float32),
        "classes": tf.cast(y["classes"], tf.float32),
    }

serving_fn(tf.random.normal([1, 640, 640, 3]))

tf.saved_model.save(
    model,
    "export/saved_model",
    signatures={"serving_default": serving_fn}
)
print('modelo exportado y guardado en export/saved_model')

'''
Luego ejecutar:
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    export/saved_model \
    tfjs_model
'''