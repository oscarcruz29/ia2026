# Proyecto 3 — Zak RNN: Autocompletado de Código con RNN en VS Code

## Descripción general

Extensión de **Visual Studio Code** que integra un modelo de lenguaje basado en una **Red Neuronal Recurrente (RNN)** entrenada sobre un corpus de código C. El modelo predice y sugiere continuaciones de código directamente en el editor, corriendo de forma completamente local sobre CPU. La arquitectura es cliente–servidor: la extensión (Node.js) se comunica con un servidor Python mediante un protocolo JSON por stdin/stdout.

---

## Arquitectura del sistema

```
VS Code (extensión JS)
        │  stdin/stdout  (JSON por línea)
        ▼
  server_stdio.py      ← Servidor RNN permanente
        │
        ▼
  model.keras          ← Vanilla RNN entrenada
        │
  meta.json            ← Vocabulario y block_size
```

El servidor Python se lanza como proceso hijo al primer uso y permanece vivo durante toda la sesión de VS Code, evitando el costo de inicialización de TensorFlow en cada solicitud.

---

## Componentes del proyecto

```
proyecto3/
├── entrenar_zak.py          # Entrenamiento del modelo RNN
├── corpus_zak.c             # Corpus de código C (estilo zak_)
├── rnn-zak/
│   ├── server_stdio.py      # Servidor de inferencia JSON/stdio
│   ├── model.keras          # Modelo exportado (generado por entrenar_zak.py)
│   └── meta.json            # Vocabulario y configuración del modelo
├── extension/
│   └── extension.js         # Extensión de VS Code (cliente Node.js)
└── venv/                    # Entorno virtual Python (requerido por TensorFlow)
```

---

## 1. Entrenamiento (`entrenar_zak.py`)

### Corpus y vocabulario

El modelo opera a nivel de **carácter**: cada carácter único del corpus es un token. El vocabulario se construye dinámicamente desde `corpus_zak.c`.

```python
chars      = sorted(set(CORPUS))
VOCAB_SIZE = len(chars)        # número de caracteres únicos
```

### Preparación de secuencias

Se generan pares `(X, Y)` con ventana deslizante de tamaño `BLOCK_SIZE = 128`:

| Variable | Forma | Descripción |
|---|---|---|
| `X` | `(N, 128)` | Secuencia de entrada |
| `Y` | `(N, 128)` | Secuencia desplazada un paso (siguiente carácter) |

### Arquitectura del modelo

```
Input(shape=(128,))
    │
Embedding(VOCAB_SIZE, 64)       ← Representa cada carácter como vector de 64 dims
    │
SimpleRNN(256, tanh, return_sequences=True)   ← Capa 1: patrones locales
    │
SimpleRNN(256, tanh, return_sequences=True)   ← Capa 2: patrones de mayor orden
    │
TimeDistributed(Dense(VOCAB_SIZE))            ← Logit por carácter, por paso
```

| Hiperparámetro | Valor |
|---|---|
| `EMBED_DIM` | 64 |
| `HIDDEN` | 256 |
| `BLOCK_SIZE` | 128 |
| `Épocas` | 150 |
| `Batch size` | 16 |
| `Optimizador` | Adam (lr=1e-3) |
| `Pérdida` | SparseCategoricalCrossentropy (from_logits=True) |

### Exportación

Al finalizar el entrenamiento se generan dos archivos en `rnn-zak/`:

- **`model.keras`** — pesos y arquitectura del modelo.
- **`meta.json`** — vocabulario (`chars`) y `block_size`, necesarios para codificar/decodificar en inferencia.

---

## 2. Servidor de inferencia (`server_stdio.py`)

Proceso Python persistente que carga el modelo una sola vez y responde peticiones JSON por stdin/stdout en un bucle infinito.

### Protocolo de comunicación

Cada mensaje es una línea JSON terminada en `\n`:

**Request:**
```json
{ "_id": 1, "method": "complete", "prefix": "int zak_", "max_new": 60, "temperature": 0.75 }
```

**Response:**
```json
{ "_id": 1, "ok": true, "text": "int zak_suma(int a, int b) {" }
```

### Métodos disponibles

| Método | Parámetros | Descripción |
|---|---|---|
| `complete` | `prefix`, `max_new`, `temperature` | Genera hasta `max_new` caracteres continuando el `prefix` |
| `suggest` | `prefix`, `n` | Devuelve hasta `n` sugerencias de completado para la línea actual |

### Lógica de generación (`_complete`)

1. Se codifica el `prefix` a índices de caracteres.
2. Se toma la ventana de los últimos `BLOCK_SIZE` tokens; si es menor, se rellena con padding por la izquierda.
3. El modelo predice logits para el siguiente carácter.
4. Se aplica **temperatura** para controlar la aleatoriedad:
   ```
   probs = softmax(logits / temperature)
   ```
5. Se muestrea el siguiente carácter y se repite hasta `max_new` veces.

### Lógica de sugerencias (`_suggest`)

Llama a `_complete` varias veces con temperatura incremental (`0.70`, `0.75`, `0.80`…), extrae el primer renglón de cada resultado, deduplica y devuelve hasta `n` opciones únicas.

### Optimización de velocidad

```python
@tf.function(reduce_retracing=True)
def predict_step(x):
    return _model(x, training=False)
```

El decorador `@tf.function` compila el paso de inferencia en un grafo TensorFlow, reduciendo significativamente la latencia de cada predicción.

---

## 3. Extensión de VS Code (`extension.js`)

Extensión escrita en **Node.js** que actúa como cliente del servidor Python.

### Gestión del proceso servidor

```javascript
proc = spawn(pythonExe, ["-u", scriptPath], {
    env: {
        PYTHONUNBUFFERED: "1",
        TF_CPP_MIN_LOG_LEVEL: "3",   // Silencia logs de TensorFlow
        CUDA_VISIBLE_DEVICES: "-1"   // Fuerza CPU
    }
});
```

- El servidor se lanza de forma **lazy** (solo al primer uso) y se reutiliza en toda la sesión.
- Se usa `readline` sobre `proc.stdout` para parsear las respuestas línea por línea.
- Las peticiones pendientes se rastrean con un `Map<id, callback>` y un timeout de **120 segundos** (necesario para el arranque inicial de TensorFlow).

### Comandos disponibles

| Comando | Atajo sugerido | Descripción |
|---|---|---|
| `zakRnn.complete` | configurable | Inserta directamente el texto generado en la posición del cursor |
| `zakRnn.suggest` | configurable | Muestra un QuickPick con hasta 5 sugerencias para elegir |

### Configuración (settings.json)

| Clave | Por defecto | Descripción |
|---|---|---|
| `zakRnn.maxNew` | `60` | Número máximo de caracteres nuevos a generar |
| `zakRnn.temperature` | `0.75` | Temperatura de muestreo (0 = determinista, 1 = más creativo) |

---

## Dependencias

### Python
```
tensorflow >= 2.10
numpy
```

### Node.js / VS Code
```
vscode API (incluida en el entorno de extensiones)
```

Configuración del entorno Python:

```bash
python -m venv venv
venv\Scripts\activate
pip install tensorflow numpy
```

---

## Flujo de uso

1. Entrenar el modelo (solo una vez):
   ```bash
   python entrenar_zak.py
   ```
2. Instalar la extensión en VS Code (modo desarrollo o `.vsix`).
3. Abrir cualquier archivo de código en el editor.
4. Posicionar el cursor y ejecutar:
   - `Zak RNN: Complete` → inserta la continuación directamente.
   - `Zak RNN: Suggest` → abre un menú de selección con variantes.
5. El servidor Python se inicia automáticamente en el primer uso y permanece activo.

---

## Notas técnicas

- El modelo opera sobre **CPU** exclusivamente (`CUDA_VISIBLE_DEVICES=-1`), eliminando dependencias de GPU.
- El timeout de 120 s en las peticiones compensa el tiempo de carga inicial de TensorFlow y el modelo.
- El servidor Python usa Python 3.11 (incompatibilidad de TensorFlow con versiones superiores).
- El corpus sigue una convención de nombres `zak_` con comentarios en español usando `//~`.
