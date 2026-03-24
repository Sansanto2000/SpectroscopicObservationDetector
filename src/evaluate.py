'''
Condigo para evaluar el rendimiento de un modelo sobre un conjunto de datos
'''
import keras_cv
import os
from utils.visualize import visualize_dataset, visualize_detections
from utils.parse import to_ragged_predictions
from utils.loadFiles import load_yolo_dataset
from utils.prepare import prepare_ds
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Ocultar mensajes de advertencia
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = '/home/sponte/Repositorios/SpectroscopicObservationDetector/models/model-v0.0.8.r.keras'
TEST_IMAGES = "/mnt/data3/sponte/datasets/conGSSSP.large.3/images"#"/Users/s.a.p.a/Documents/Datasets/conGSSSP/images/" # "D:\\Datasets\\conGSSSP_v2\\images\\" 
TEST_ANNOT = "/mnt/data3/sponte/datasets/conGSSSP.large.3/labels"
# TEST_IMAGES = "/mnt/data3/sponte/datasets/observaciones-etiquetadas/images.jpg"
# TEST_ANNOT = "/mnt/data3/sponte/datasets/observaciones-etiquetadas/labels"
ROTATE_ANGLE = 0
SPLIT_RATIO = 0.2

# Etiquetas de clase
class_ids = [
    "observation"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Datos
train_data, test_data = load_yolo_dataset(
    TEST_IMAGES, 
    TEST_ANNOT, 
    SPLIT_RATIO, 
    int(os.getenv("RANDOM_SEED")))

# Preparar datos de test
test_ds = prepare_ds(test_data, (640,640), int(os.getenv("BATCH_SIZE")), rotate_angle=ROTATE_ANGLE)

# Visualizar datos
visualize_dataset(
    test_ds, 
    bounding_box_format="rel_xywh", 
    value_range=(0, 255), 
    rows=4, 
    cols=4, 
    class_mapping=class_mapping, 
    save_path=os.path.join(os.getenv("PLOT_PATH"), "test_visualization.png"),
)

# Tuplas
def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]
test_ds = test_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

### Modelo ###
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

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
    rows=4, cols=4, class_mapping=class_mapping, 
    save_path=os.path.join(os.getenv("PLOT_PATH"), "test_predictions.png"),
    confidence_threshold=0.5
)