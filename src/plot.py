'''
Condigo para evaluar el rendimiento de un modelo sobre un conjunto de datos
'''
import keras_cv
import os
from utils.visualize import print_dataset_info_dict, visualize_dataset, visualize_detections
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
ROTATE_ANGLE = 90
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

print_dataset_info_dict(test_ds, "Validation")

batch_size = int(os.getenv("BATCH_SIZE"))
num_batches = tf.data.experimental.cardinality(test_ds).numpy()
num_images = num_batches * batch_size
print("Cantidad aproximada de imágenes:", num_images)

# Visualizar datos
visualize_dataset(
    test_ds, 
    bounding_box_format="rel_xywh", 
    value_range=(0, 255), 
    rows=2, 
    cols=3, 
    class_mapping=class_mapping, 
    save_path=os.path.join(os.getenv("PLOT_PATH"), "plot_real_labeled.png"),
)