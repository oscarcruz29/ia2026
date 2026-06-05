# Proyecto 4 — Tutor Analítico RAG: Fine-tuning + Recuperación de Documentos sobre Violencia en México

## Descripción general

Sistema de pregunta-respuesta especializado en **seguridad pública mexicana**, construido sobre dos pilares:

1. **Fine-tuning con LoRA** de un LLM base (Llama 3.2 3B) sobre un dataset curado de preguntas y respuestas del dominio.
2. **RAG (Retrieval-Augmented Generation)** con índice vectorial FAISS para anclar las respuestas en documentos reales (PDF, TXT, MD).

El sistema incluye dos variantes de inferencia: una con modelo **Ollama local** (`rag_ollama.py`) y otra con el **modelo fine-tuneado** mediante adaptadores LoRA (`rag_lora.py`), más un enrutador inteligente que distingue preguntas sobre el corpus de preguntas sobre su contenido.

---

## Arquitectura general

```
Usuario (pregunta CLI)
        │
        ▼
  Enrutador de preguntas
   ├── Meta-pregunta?  ──► build_meta_prompt()  ──► Respuesta estructural
   └── Normal?         ──► FAISS search (top-k)
                               │
                        Chunks recuperados
                               │
                        build_prompt() (RAG)
                               │
                     LLM (Ollama o LoRA local)
                               │
                        RESPUESTA citada
```

---

## Componentes del proyecto

```
proyecto4/
├── fine_tuning.ipynb / train.py         # Entrenamiento LoRA del modelo
├── rag_ollama.py                         # Inferencia RAG con Ollama (versión base)
├── rag_lora.py                           # Inferencia RAG con modelo fine-tuneado
├── dataset_entrenamiento.jsonl           # Dataset de entrenamiento (formato ChatML)
├── tutor_analitico_lora/                 # Pesos LoRA exportados
└── docs/                                 # Corpus de documentos
    ├── documento1.pdf
    ├── ...
    ├── vector_index.faiss                # Índice vectorial persistido
    └── chunks_metadata.pkl              # Metadatos de chunks persistidos
```

---

## 1. Fine-tuning con LoRA

### Modelo base

```
unsloth/Llama-3.2-3B-Instruct
```

Modelo de lenguaje causal en 16-bit (`torch.float16`), optimizado para GPU T4 via `device_map="auto"`.

### Dataset

Archivo `dataset_entrenamiento.jsonl` con conversaciones en formato ChatML (`messages`). Se formatea con `apply_chat_template` del tokenizador de Llama 3 antes del entrenamiento.

### Configuración LoRA

```python
LoraConfig(
    r              = 16,
    lora_alpha     = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM
)
```

LoRA inyecta matrices de bajo rango en las capas de atención y feed-forward, entrenando solo una fracción pequeña de los parámetros totales del modelo.

| Parámetro | Valor |
|---|---|
| `r` (rango) | 16 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| Módulos objetivo | q, k, v, o, gate, up, down proj |

### Hiperparámetros de entrenamiento

| Parámetro | Valor |
|---|---|
| `max_steps` | 60 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 (batch efectivo = 8) |
| `learning_rate` | 2e-4 |
| `max_length` | 2048 tokens |
| `fp16` | True |
| `optimizer` | adamw_torch |
| `gradient_checkpointing` | True (reduce uso de VRAM) |

### Exportación

```
tutor_analitico_lora/     ← adaptadores LoRA listos para inferencia
```

En inferencia los pesos LoRA se fusionan al modelo base con `merge_and_unload()` para mayor estabilidad.

---

## 2. Pipeline de documentos (compartido por ambas variantes RAG)

### Formatos soportados

`.pdf`, `.txt`, `.md` — detectados recursivamente en el directorio de documentos.

### Extracción de texto

Los PDFs se procesan con **PyMuPDF** (`fitz`), extrayendo texto página por página. Los archivos de texto plano se leen con UTF-8.

### Chunking semántico

```python
smart_chunk_text(text, chunk_chars=1500, overlap_chars=250)
```

En lugar de cortar en posición fija, el algoritmo retrocede hasta el último `.` o `\n` dentro de la segunda mitad del chunk para no romper oraciones:

```
|────────── chunk_chars ──────────|
                   |──── zona de corte inteligente ────|
                              ↑ último "." o "\n"
```

El **overlap** de 250 caracteres entre chunks consecutivos preserva el contexto en los bordes.

### Embeddings vectoriales

Los embeddings se generan con el modelo `nomic-embed-text` mediante la API local de **Ollama**:

```
POST http://localhost:11434/api/embeddings
```

Los vectores se normalizan L2 antes de indexar para usar **similitud coseno** vía producto punto interno (`IndexFlatIP`).

### Persistencia del índice

El índice FAISS y los metadatos de chunks se guardan en disco tras la primera construcción:

```
vector_index.faiss      ← vectores normalizados
chunks_metadata.pkl     ← lista de objetos Chunk
```

En ejecuciones posteriores se cargan directamente (flag `--force-rebuild` para forzar reconstrucción).

---

## 3. Sistema de preguntas

### Enrutador inteligente

```python
if es_meta_pregunta(question):
    prompt = build_meta_prompt(question, chunks)  # Respuesta estructural
else:
    # Búsqueda FAISS → RAG normal
    prompt = build_prompt(question, retrieved)
```

**Meta-preguntas** son consultas sobre la *estructura* del corpus (cuántos documentos, qué autores, qué instituciones aparecen), detectadas por palabras clave. Se responden sin búsqueda vectorial, usando metadatos directos de los chunks.

**Preguntas normales** pasan por el pipeline RAG completo.

### Extracción de autores e instituciones

El módulo `extraer_autores()` usa expresiones regulares para identificar:

- **Autores personales**: patrones de nombre propio con formato APA y frases introductorias (`elaborado por`, `coordinado por`…).
- **Instituciones y dependencias**: siglas conocidas (INEGI, SESNSP, FGR, SEDENA…) y patrones de `Secretaría de`, `Comisión`, `Instituto`, `Centro`.

### Prompt RAG estándar

El prompt usa el formato de chat nativo de Llama 3 (`<|begin_of_text|>`, `<|start_header_id|>`…) e instruye al modelo a:

- Responder **solo** con el contexto recuperado.
- Citar siempre el documento fuente al final de cada afirmación.
- Listar las fuentes consultadas en una sección final.

---

## 4. Comparativa de variantes

| Característica | `rag_ollama.py` | `rag_lora.py` |
|---|---|---|
| Modelo generativo | Ollama (`llama3.2:3b`) | Llama 3.2 3B + LoRA fine-tuneado |
| Requiere Ollama activo | Sí | Solo para embeddings |
| GPU necesaria | No | Opcional (funciona en CPU con float32) |
| Enrutador meta-preguntas | No | Sí |
| Extracción de autores/inst. | No | Sí |
| Formato de prompt | Texto libre | ChatML nativo de Llama 3 |

---

## Dependencias

```
torch
transformers
peft
trl
datasets
faiss-cpu
pymupdf          # fitz
requests
tqdm
numpy
```

Instalación:

```bash
pip install torch transformers peft trl datasets faiss-cpu pymupdf requests tqdm numpy
```

También se requiere **Ollama** corriendo localmente para los embeddings:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b   # Solo para rag_ollama.py
```

---

## Uso

### Entrenar el modelo fine-tuneado

```bash
# En Jupyter / Colab (GPU T4 recomendada)
# Ejecutar las celdas de fine_tuning.ipynb
# Salida: carpeta tutor_analitico_lora/
```

### Inferencia con Ollama (sin fine-tuning)

```bash
python rag_ollama.py \
  --docs ./docs \
  --question "¿Cuál es la tasa de homicidios en Michoacán?"
```

### Inferencia con modelo fine-tuneado

```bash
python rag_lora.py \
  --docs ./docs \
  --lora-path ./tutor_analitico_lora \
  --question "¿Qué instituciones menciona el corpus sobre seguridad pública?"
```

### Parámetros comunes

| Flag | Por defecto | Descripción |
|---|---|---|
| `--docs` | requerido | Directorio con documentos PDF/TXT/MD |
| `--question` | requerido | Pregunta del usuario |
| `--topk` | 4 | Número de chunks a recuperar |
| `--chunk-chars` | 1500 | Tamaño máximo de chunk en caracteres |
| `--overlap-chars` | 250 | Solapamiento entre chunks |
| `--embed-model` | `nomic-embed-text` | Modelo de embeddings en Ollama |
| `--ollama-url` | `http://localhost:11434` | URL de la instancia de Ollama |
| `--force-rebuild` | False | Reconstruye el índice FAISS desde cero |

---

## Flujo de uso recomendado

1. Colocar los documentos PDF/TXT/MD en `./docs/`.
2. *(Opcional)* Entrenar el modelo en Colab con GPU y descargar `tutor_analitico_lora/`.
3. Lanzar Ollama localmente y descargar el modelo de embeddings.
4. Primera ejecución: el índice FAISS se construye y persiste automáticamente.
5. Ejecuciones siguientes: el índice se carga desde disco (respuesta más rápida).
