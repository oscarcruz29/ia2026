import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# ─── 1. Importar el modelo entrenado ──────────────────────────────────────────
ruta_modelo = "modelo_animales.keras"

try:
    modelo_cargado = keras.models.load_model(ruta_modelo)
    print(f"✓ Modelo '{ruta_modelo}' cargado exitosamente.\n")
except Exception as e:
    print(f"❌ Error al cargar el modelo. Verifica que la ruta sea correcta.\nDetalles: {e}")
    exit()

# ─── 2. Configuración ─────────────────────────────────────────────────────────
IMG_H, IMG_W = 128, 128

animales = ['arana', 'ballena', 'mono', 'pajaro', 'rana'] 

# ─── 3. Función de Predicción ─────────────────────────────────────────────────
def predecir(ruta_imagen):
    print("-" * 50)
    print(f"Procesando: {ruta_imagen}")
    
    try:
        img = image.load_img(ruta_imagen, target_size=(IMG_H, IMG_W))
        
        img_array = image.img_to_array(img)
        
        img_array = tf.expand_dims(img_array, 0)
        
        # Normalizar los píxeles al rango [0, 1] (igual que en el entrenamiento)
        img_array /= 255.0
        
        predicciones = modelo_cargado.predict(img_array, verbose=0)
        
        indice_clase = np.argmax(predicciones[0])
        confianza = np.max(predicciones[0]) * 100
        clase_predicha = animales[indice_clase]
        
        print(f"Predicción: ¡Es un(a) {clase_predicha.upper()}!")
        print(f"Confianza:  {confianza:.2f}%")
        
        
    except FileNotFoundError:
        print("❌ Error: No se encontró la imagen en la ruta especificada.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

# ─── 4. Pruebas ───────────────────────────────────────────────────────────────
print("Iniciando pruebas de predicción...\n")

predecir('/Users/Leo/Downloads/ballena-1_1b15f788_221212154535_1280x720.jpg')
predecir('//Users/Leo/Downloads/hero-articulo-pajaro-o-ave-1200.jpg.webp')
predecir('/Users/Leo/Downloads/images.jpeg')
predecir('/Users/Leo/Downloads/11295483124-acaf5d0519-k.jpg')
predecir('/Users/Leo/Downloads/Ateles_fusciceps_Colombia.JPG')
