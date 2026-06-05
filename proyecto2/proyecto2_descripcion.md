# Clasificador de Animales con CNN (Keras / TensorFlow)

## Descripción general

Script de entrenamiento en **Python** con **TensorFlow / Keras** que construye y entrena una **Red Neuronal Convolucional (CNN)** para clasificar imágenes de animales en múltiples categorías. El modelo carga automáticamente el dataset desde disco, aplica aumento de datos, entrena durante 50 épocas y exporta el modelo listo para inferencia.

---

## Configuración principal

| Parámetro | Valor | Descripción |
|---|---|---|
| `IMG_H` / `IMG_W` | 128 × 128 px | Tamaño al que se redimensionan todas las imágenes |
| `BATCH_SIZE` | 32 | Imágenes por lote de entrenamiento |
| `EPOCHS` | 50 | Épocas de entrenamiento |
| `INIT_LR` | 1e-4 | Learning rate inicial para Adam |
| `DATASET_DIR` | `./dataset/` | Directorio raíz del dataset |

---

## Pipeline de datos

### Carga

Las imágenes se leen directamente desde disco usando `image_dataset_from_directory`, que infiere automáticamente las clases a partir de la estructura de carpetas:

```
dataset/
├── clase_1/
│   ├── imagen_a.jpg
│   └── ...
├── clase_2/
│   └── ...
└── clase_N/
    └── ...
```

La división se realiza con semilla fija (`seed=48`):

| Split | Fracción | Uso |
|---|---|---|
| `train_ds` | 80 % | Entrenamiento del modelo |
| `val_ds` | 20 % | Evaluación y monitoreo |

Las etiquetas se generan en formato **one-hot** (`label_mode='categorical'`).

### Normalización

Los píxeles se escalan de `[0, 255]` a `[0.0, 1.0]` mediante una capa `Rescaling(1./255)` aplicada con `.map()`.

### Optimización de carga

```python
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

- `.cache()` almacena el dataset en memoria tras la primera época, eliminando re-lecturas de disco.
- `.prefetch(AUTOTUNE)` solapa la preparación del siguiente lote con el paso de entrenamiento actual.

---

## Arquitectura del modelo

```
Sequential
├── Data Augmentation
│   ├── RandomFlip("horizontal")
│   ├── RandomRotation(0.1)
│   └── RandomZoom(0.1)
│
├── Bloque Conv 1
│   ├── Conv2D(32, 3×3, padding='same')
│   ├── LeakyReLU(α=0.1)
│   ├── MaxPooling2D(2×2)
│   └── Dropout(0.3)
│
├── Bloque Conv 2
│   ├── Conv2D(64, 3×3, padding='same')
│   ├── LeakyReLU(α=0.1)
│   ├── MaxPooling2D(2×2)
│   └── Dropout(0.3)
│
└── Clasificador
    ├── Flatten()
    ├── Dense(64)
    ├── LeakyReLU(α=0.1)
    ├── Dropout(0.3)
    └── Dense(nClasses, softmax)
```

### Decisiones de diseño

| Elemento | Justificación |
|---|---|
| **LeakyReLU** (α=0.1) en lugar de ReLU | Evita el problema de "neuronas muertas" al permitir gradientes pequeños para activaciones negativas |
| **Dropout(0.3)** tras cada bloque | Regularización para reducir sobreajuste |
| **Data Augmentation integrado** | Al ser capas del modelo, el aumento solo se aplica durante entrenamiento (no en inferencia) |
| **Softmax** en la capa de salida | Genera una distribución de probabilidad sobre todas las clases |
| **padding='same'** | Conserva las dimensiones espaciales tras cada convolución |

---

## Compilación y entrenamiento

```python
animal_model.compile(
    loss      = CategoricalCrossentropy(),
    optimizer = Adam(learning_rate=1e-4),
    metrics   = ['accuracy']
)

history = animal_model.fit(
    train_ds,
    epochs          = 50,
    validation_data = val_ds,
    verbose         = 1
)
```

- **Función de pérdida**: `CategoricalCrossentropy`, adecuada para clasificación multiclase con etiquetas one-hot.
- **Optimizador**: Adam con LR fijo de `1e-4`.
- El objeto `history` contiene las curvas de `loss` y `accuracy` por época para ambos splits.

---

## Salida del script

Al finalizar el entrenamiento el script imprime las métricas finales en el set de validación:

```
Validation Loss:     X.XXXX
Validation Accuracy: X.XXXX
```

Y guarda el modelo entrenado en disco:

```
✓ Modelo guardado exitosamente como 'modelo_animales.keras'
```

El archivo `.keras` contiene la arquitectura, los pesos y la configuración del optimizador, y puede cargarse directamente para inferencia:

```python
modelo = tf.keras.models.load_model("modelo_animales.keras")
prediccion = modelo.predict(imagen)
```

---

## Estructura de archivos

```
proyecto/
├── train.py                  # Script principal de entrenamiento
├── modelo_animales.keras     # Modelo exportado (generado al correr el script)
└── dataset/
    ├── clase_1/
    ├── clase_2/
    └── clase_N/
```

---

## Dependencias

```
tensorflow >= 2.10
```

Instalación:

```bash
pip install tensorflow
```

---

## Flujo de uso

1. Organizar el dataset en subcarpetas por clase dentro de `./dataset/`.
2. Ejecutar el script:
   ```bash
   python train.py
   ```
3. Al finalizar, usar `modelo_animales.keras` para inferencia en producción.
