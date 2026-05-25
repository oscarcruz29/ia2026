#!/usr/bin/env python3
"""Servidor RNN zak — JSON por linea en stdin/stdout."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent

_model  = None
_stoi: dict[str, int] = {}
_itos: dict[int, str] = {}
_BLOCK  = 64
_fast_predict = None # Agregamos una variable global para el compilador

def _load():
    global _model, _stoi, _itos, _BLOCK, _fast_predict
    meta    = json.loads((ROOT / "meta.json").read_text(encoding="utf-8"))
    _BLOCK  = int(meta["block_size"])
    chars   = meta["chars"]
    _stoi   = {c: i for i, c in enumerate(chars)}
    _itos   = {i: c for c, i in _stoi.items()}
    _model  = tf.keras.models.load_model(ROOT / "model.keras")

    # --- OPTIMIZACIÓN CRUCIAL PARA VELOCIDAD ---
    @tf.function(reduce_retracing=True)
    def predict_step(x):
        return _model(x, training=False)
    
    _fast_predict = predict_step

def _enc(s):  return [_stoi[c] for c in s if c in _stoi]
def _dec(ids): return "".join(_itos[i] for i in ids)

def _complete(prefix, max_new=80, temperature=0.75):
    ids = _enc(prefix)
    if not ids:
        return prefix
    rng = np.random.default_rng(42)
    for _ in range(max_new):
        x = np.array(ids[-_BLOCK:], dtype=np.int64)
        if x.shape[0] < _BLOCK:
            pad = np.full(_BLOCK - x.shape[0], ids[0], dtype=np.int64)
            x   = np.concatenate([pad, x])
        
        # --- USAMOS LA FUNCIÓN COMPILADA ---
        logits = _fast_predict(x.reshape(1, _BLOCK)).numpy()[0, -1, :]
        
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()
        probs  = np.exp(logits)
        probs  = probs / probs.sum()
        ids.append(int(rng.choice(len(probs), p=probs)))
    return _dec(ids)

def _suggest(prefix, n=5):
    seen, out = set(), []
    for i in range(n * 3):
        text = _complete(prefix, max_new=50, temperature=0.7 + 0.05 * i)
        line = (prefix + text[len(prefix):].split("\n")[0])[:80]
        if line not in seen and len(line) > len(prefix):
            seen.add(line)
            out.append(line)
        if len(out) >= n:
            break
    return out

def _handle(msg):
    rid  = msg.get("_id")
    base = {"_id": rid} if rid is not None else {}
    try:
        if msg.get("method") == "complete":
            return {**base, "ok": True,
                    "text": _complete(msg.get("prefix",""),
                                      int(msg.get("max_new", 80)),
                                      float(msg.get("temperature", 0.75)))}
        if msg.get("method") == "suggest":
            return {**base, "ok": True,
                    "items": _suggest(msg.get("prefix",""), int(msg.get("n", 5)))}
        return {**base, "ok": False, "error": "metodo desconocido"}
    except Exception as exc:
        return {**base, "ok": False, "error": str(exc)}

def main():
    if not (ROOT / "model.keras").is_file():
        sys.stderr.write("Falta model.keras — ejecuta entrenar_zak.py primero.\n")
        sys.exit(1)
    _load()
    sys.stderr.write("servidor zak listo\n")
    sys.stderr.flush()
    
    # --- SOLUCIÓN AL BÚFER DE ENTRADA ---
    while True:
        line = sys.stdin.readline()
        if not line: # Cierre de pipe o EOF
            break
        line = line.strip()
        if not line:
            continue
        try:
            out = _handle(json.loads(line))
        except json.JSONDecodeError as exc:
            out = {"ok": False, "error": f"JSON invalido: {exc}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()