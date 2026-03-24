import os
from PIL import Image

# Carpeta de entrada y salida
input_folder = "/mnt/data3/sponte/datasets/observaciones-etiquetadas/images"
output_folder = "/mnt/data3/sponte/datasets/observaciones-etiquetadas/images.jpg"

# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Iterar por todos los archivos PNG
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        png_path = os.path.join(input_folder, filename)
        # Abrir imagen
        img = Image.open(png_path).convert("RGB")  # convertir a RGB
        # Crear path de salida cambiando extensión
        jpg_filename = os.path.splitext(filename)[0] + ".jpg"
        jpg_path = os.path.join(output_folder, jpg_filename)
        # Guardar como JPG
        img.save(jpg_path, "JPEG", quality=95)

print("Conversión completa!")