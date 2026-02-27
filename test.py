import os
import tensorflow as tf

image_dir ="/mnt/data3/sponte/datasets/conGSSSP.large/images"


cant = 0
for root, _, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(".jpg"):
            path = os.path.join(root, file)
            try:
                img = tf.io.read_file(path)
                tf.io.decode_jpeg(img)
            except Exception as e:
                print("Imagen corrupta:", path)
                cant += 1


print('total corruptas ', cant)