import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU

# ─── 1. Configuración ─────────────────────────────────────────────────────────
IMG_H, IMG_W = 128, 128
BATCH_SIZE   = 32
EPOCHS       = 50
INIT_LR      = 1e-4
DATASET_DIR  = os.path.join(os.getcwd(), 'dataset')

print(f"Cargando imágenes desde: {DATASET_DIR}")

# ─── 2. Carga de Datos (Eficiente en Memoria) ─────────────────────────────────
# Crea el dataset de entrenamiento (80%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=48,
    image_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Crea el dataset de validación (20%)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=48,
    image_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

animales = train_ds.class_names
nClasses = len(animales)
print('\nClases detectadas:', animales)
print('Total de clases:', nClasses)

# Normalización (escala los píxeles de 0-255 a 0.0-1.0)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Optimización de rendimiento para la carga de datos
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ─── 3. Construcción del Modelo ───────────────────────────────────────────────
animal_model = Sequential([
    # Data Augmentation integrado
    keras.layers.RandomFlip("horizontal", input_shape=(IMG_H, IMG_W, 3)),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),

    # Capas Convolucionales
    Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.3),

    Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.3),

    # Clasificador
    Flatten(),
    Dense(64, activation='linear'),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(nClasses, activation='softmax')
])

animal_model.summary()

animal_model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=INIT_LR),
    metrics=['accuracy']
)

# ─── 4. Entrenamiento y Exportación ───────────────────────────────────────────
print("\nIniciando entrenamiento...")
history = animal_model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    verbose=1
)

# Evaluar métricas finales en el set de validación
print("\nEvaluando modelo en el set de validación...")
val_loss, val_acc = animal_model.evaluate(val_ds, verbose=0)
print(f'Validation Loss:     {val_loss:.4f}')
print(f'Validation Accuracy: {val_acc:.4f}\n')

# Guardar el modelo en disco
animal_model.save("modelo_animales.keras")
print("✓ Modelo guardado exitosamente como 'modelo_animales.keras'")