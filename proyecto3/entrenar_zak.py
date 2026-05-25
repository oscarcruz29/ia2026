import json
from pathlib import Path
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

CORPUS = Path("corpus_zak.c").read_text(encoding="utf-8")

chars = sorted(set(CORPUS))
stoi  = {ch: i for i, ch in enumerate(chars)}
itos  = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(chars)

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(ids):
    return "".join(itos[i] for i in ids)

SEQ = np.array(encode(CORPUS), dtype=np.int64)
print("VOCAB_SIZE:", VOCAB_SIZE, "| chars:", len(CORPUS))

BLOCK_SIZE = 128

X_rows, Y_rows = [], []
for i in range(0, len(SEQ) - BLOCK_SIZE):
    X_rows.append(SEQ[i : i + BLOCK_SIZE])
    Y_rows.append(SEQ[i + 1 : i + 1 + BLOCK_SIZE])

X = np.stack(X_rows)
Y = np.stack(Y_rows)
print("X:", X.shape, "Y:", Y.shape)

EMBED_DIM = 64
HIDDEN    = 128

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(BLOCK_SIZE,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM),
    tf.keras.layers.SimpleRNN(HIDDEN, activation="tanh", return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(VOCAB_SIZE)),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.summary()

history = model.fit(X, Y, epochs=150, batch_size=32, verbose=1)

print("perdida inicial:", round(float(history.history["loss"][0]),  4))
print("perdida final:  ", round(float(history.history["loss"][-1]), 4))

def complete(prompt, max_new=120, temperature=0.75):
    ids = encode(prompt)
    rng = np.random.default_rng(42)
    for _ in range(max_new):
        x = np.array(ids[-BLOCK_SIZE:], dtype=np.int64)
        if x.shape[0] < BLOCK_SIZE:
            pad = np.full(BLOCK_SIZE - x.shape[0], ids[0], dtype=np.int64)
            x   = np.concatenate([pad, x])
        logits = model(x.reshape(1, BLOCK_SIZE), training=False).numpy()[0, -1, :]
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()
        probs  = np.exp(logits)
        probs  = probs / probs.sum()
        ids.append(int(rng.choice(len(probs), p=probs)))
    return decode(ids)

print(complete("int zak_", max_new=80, temperature=0.7))

DEPLOY = Path("rnn-zak")
DEPLOY.mkdir(parents=True, exist_ok=True)
model.save(DEPLOY / "model.keras")
(DEPLOY / "meta.json").write_text(
    json.dumps({"block_size": BLOCK_SIZE, "chars": chars}, ensure_ascii=False),
    encoding="utf-8",
)
print("Guardado en:", DEPLOY.resolve())