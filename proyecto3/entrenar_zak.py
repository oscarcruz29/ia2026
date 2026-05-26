import json
from pathlib import Path
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

# 1. Cargamos tu corpus real de C
CORPUS = Path("corpus_zak.c").read_text(encoding="utf-8")

chars = sorted(set(CORPUS))
stoi  = {ch: i for i, ch in enumerate(chars)}
itos  = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(chars)

def encode(s):
    return [stoi[c] for c in s if c in stoi]

SEQ = np.array(encode(CORPUS), dtype=np.int64)
print("Tamaño del Vocabulario:", VOCAB_SIZE, "| Total de caracteres:", len(CORPUS))

# 2. Ventana grande para recordar abrir y cerrar llaves {}
BLOCK_SIZE = 128

X_rows, Y_rows = [], []
for i in range(0, len(SEQ) - BLOCK_SIZE):
    X_rows.append(SEQ[i : i + BLOCK_SIZE])
    Y_rows.append(SEQ[i + 1 : i + 1 + BLOCK_SIZE])

X = np.stack(X_rows)
Y = np.stack(Y_rows)

# 3. Arquitectura más profunda (Apilando SimpleRNN)
EMBED_DIM = 64
HIDDEN    = 256

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(BLOCK_SIZE,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM),
    
    # Dos capas para que entienda patrones complejos
    tf.keras.layers.SimpleRNN(HIDDEN, activation="tanh", return_sequences=True),
    tf.keras.layers.SimpleRNN(HIDDEN, activation="tanh", return_sequences=True),
    
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(VOCAB_SIZE)),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.summary()

# 4. Entrenamiento (Batch 16 ayuda a generalizar mejor)
model.fit(X, Y, epochs=150, batch_size=16, verbose=1)

# 5. Guardamos el modelo para que la extensión de VS Code lo consuma
DEPLOY = Path("rnn-zak")
DEPLOY.mkdir(parents=True, exist_ok=True)
model.save(DEPLOY / "model.keras")

(DEPLOY / "meta.json").write_text(
    json.dumps({"block_size": BLOCK_SIZE, "chars": chars}, ensure_ascii=False),
    encoding="utf-8",
)
print("Modelo guardado exitosamente en:", DEPLOY.resolve())