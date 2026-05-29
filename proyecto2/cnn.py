import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from keras.layers import LeakyReLU

# ─── Cargar imágenes ──────────────────────────────────────────────────────────
# Estructura esperada:
#   dataset/
#     ballena/   imagen1.jpg ...
#     pajaro/    imagen1.jpg ...
#     arana/     imagen1.jpg ...
#     mono/      imagen1.jpg ...

IMG_H, IMG_W = 128, 128
DATASET_DIR  = os.path.join(os.getcwd(), 'dataset')

images      = []
directories = []
dircount    = []
prevRoot    = ''
cant        = 0

print("Leyendo imágenes de", DATASET_DIR)

for root, dirnames, filenames in os.walk(DATASET_DIR):
    for filename in filenames:
        if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant += 1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            if len(image.shape) == 3:
                images.append(image)
            if prevRoot != root:
                prevRoot = root
                directories.append(root)
                dircount.append(cant)
                cant = 0

dircount.append(cant)
dircount = dircount[1:]
dircount[0] = dircount[0] + 1

print('Directorios leídos:', len(directories))
print('Imágenes en cada directorio:', dircount)
print('Total de imágenes:', sum(dircount))

# ─── Etiquetas ────────────────────────────────────────────────────────────────
labels = []
indice = 0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice += 1

print("Etiquetas creadas:", len(labels))

animales = []
for directorio in directories:
    name = directorio.split(os.sep)
    animales.append(name[-1])
    print(animales.index(name[-1]), name[-1])

# ─── Numpy arrays ─────────────────────────────────────────────────────────────
y = np.array(labels)
X = np.array(images, dtype=np.uint8)

classes  = np.unique(y)
nClasses = len(classes)
print('Total de clases:', nClasses)
print('Clases:', classes)

# ─── Train / Test split ───────────────────────────────────────────────────────
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2)
print('Training shape:', train_X.shape, train_Y.shape)
print('Testing shape :', test_X.shape,  test_Y.shape)

# ─── Normalización ────────────────────────────────────────────────────────────
train_X = train_X.astype('float32') / 255.
test_X  = test_X.astype('float32')  / 255.

# ─── One-hot encoding ─────────────────────────────────────────────────────────
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot  = to_categorical(test_Y)

print('Etiqueta original:', train_Y[0])
print('One-hot:', train_Y_one_hot[0])

# ─── Train / Validation split ─────────────────────────────────────────────────
train_X, valid_X, train_label, valid_label = train_test_split(
    train_X, train_Y_one_hot, test_size=0.2, random_state=48
)
print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

# ─── Modelo ───────────────────────────────────────────────────────────────────
INIT_LR    = 1e-4
epochs     = 50
batch_size = 32

animal_model = Sequential()
animal_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
                        padding='same', input_shape=(IMG_H, IMG_W, 3)))
animal_model.add(LeakyReLU(alpha=0.1))
animal_model.add(MaxPooling2D((2, 2), padding='same'))
animal_model.add(Dropout(0.3))

animal_model.add(Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same'))
animal_model.add(LeakyReLU(alpha=0.1))
animal_model.add(MaxPooling2D((2, 2), padding='same'))
animal_model.add(Dropout(0.3))

animal_model.add(Flatten())
animal_model.add(Dense(64, activation='linear'))
animal_model.add(LeakyReLU(alpha=0.1))
animal_model.add(Dropout(0.3))
animal_model.add(Dense(nClasses, activation='softmax'))

animal_model.summary()

animal_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
    metrics=['accuracy']
)

# ─── Entrenamiento ────────────────────────────────────────────────────────────
animal_train = animal_model.fit(
    train_X, train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label)
)

animal_model.save("modelo_animales.keras")

# ─── Evaluación ───────────────────────────────────────────────────────────────
test_eval = animal_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Test loss:    ', test_eval[0])
print('Test accuracy:', test_eval[1])

# ─── Reporte de clasificación ─────────────────────────────────────────────────
predicted_classes2 = animal_model.predict(test_X)

predicted_classes = []
for pred in predicted_classes2:
    predicted_classes.append(pred.tolist().index(max(pred)))
predicted_classes = np.array(predicted_classes)

print(classification_report(test_Y, predicted_classes, target_names=animales))

# ─── Predecir imagen nueva ────────────────────────────────────────────────────
from skimage.transform import resize

def predecir(ruta_imagen):
    image = plt.imread(ruta_imagen)
    image_resized = resize(image, (IMG_H, IMG_W),
                           anti_aliasing=True, clip=False, preserve_range=True)
    X_new  = np.array([image_resized], dtype=np.uint8)
    X_new  = X_new.astype('float32') / 255.
    pred   = animal_model.predict(X_new)
    clase  = animales[pred[0].tolist().index(max(pred[0]))]
    print(ruta_imagen, "→", clase)

predecir('/Users/Leo/Downloads/ballena-1_1b15f788_221212154535_1280x720.jpg')
predecir('/Users/Leo/Downloads/png1.jpg')
predecir('/Users/Leo/Downloads/images.jpeg')
# Ejemplo de uso:
# predecir('/ruta/a/tu/imagen.jpg')