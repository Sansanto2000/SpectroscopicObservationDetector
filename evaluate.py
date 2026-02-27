import keras_cv
import os
from utils.visualize import visualize_dataset, visualize_detections
from utils.parse import to_ragged_predictions
from utils.loadFiles import load_yolo_dataset
from utils.prepare import prepare_ds
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Ocultar mensajes de advertencia

TEST_IMAGES = "/mnt/data3/sponte/datasets/observaciones-etiquetadas/images.jpg"
TEST_ANNOT = "/mnt/data3/sponte/datasets/observaciones-etiquetadas/labels"
BATCH_SIZE = 4

# Etiquetas de clase
class_ids = [
    "observation"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Datos
test_data, _ = load_yolo_dataset(TEST_IMAGES, TEST_ANNOT, 0)
# Preparar datos de entrenamiento
test_ds = prepare_ds(test_data, (640,640), BATCH_SIZE, rotate_angle=90)

# Visualizar datos
visualize_dataset(
    test_ds, 
    bounding_box_format="rel_xywh", 
    value_range=(0, 255), 
    rows=2, 
    cols=2, 
    class_mapping=class_mapping, 
    save_path="train/test_visualization.png"
)

# Tuplas
def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]
test_ds = test_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

### Modelo ###
model = tf.keras.models.load_model(
    '/home/sponte/Repositorios/SpectroscopicObservationDetector/models/model.keras',
    compile=False
)

### Metricas ###
metrics = keras_cv.metrics.BoxCOCOMetrics(
    bounding_box_format="xywh",
    evaluate_freq=1e9,
)

### Predecir y evaluar ###
for images, y_true in test_ds:
    y_pred = model.predict(images, verbose=0)
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
    metrics.update_state(y_true, y_pred)

metrics = metrics.result(force=True)
print(metrics)

### Visualizacion ###
# Predicciones
visualize_detections(
    model=model, 
    dataset=test_ds, 
    bounding_box_format="rel_xywh",
    rows=2, cols=2, class_mapping=class_mapping, 
    save_path="train/test_predictions.png",
    confidence_threshold=0.5
)