import argparse
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz
import numpy as np
import requests
from tqdm import tqdm
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    path: str
    text: str
    start_char: int

# ─── Carga del modelo fine-tuneado ────────────────────────────────────────────
def load_finetuned_model(lora_path: str, base_model_id: str = "unsloth/Llama-3.2-3B-Instruct"):
    print(f"Cargando modelo base: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map=None,           # ← quitamos device_map="auto"
        dtype=torch.float32,       # ← float32 para CPU
        low_cpu_mem_usage=True,
    )

    print(f"Cargando adaptadores LoRA desde: {lora_path}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        is_trainable=False,
    )
    model = model.merge_and_unload()  # ← fusiona LoRA al base, más estable
    model.eval()

    # Detectar si hay GPU disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,  # 0=GPU, -1=CPU
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
    )
    print("✅ Modelo fine-tuneado listo.")
    return pipe

# ─── Generación local ─────────────────────────────────────────────────────────
def local_generate(prompt: str, pipe) -> str:
    output = pipe(prompt)
    full_text = output[0]["generated_text"]
    if prompt in full_text:
        return full_text[len(prompt):].strip()
    return full_text.strip()

# ─── Detección de meta-preguntas ──────────────────────────────────────────────
def es_meta_pregunta(question: str) -> bool:
    keywords = [
        "cuántos corpus", "cuantos corpus",
        "cuántos documentos", "cuantos documentos",
        "cuántos autores", "cuantos autores",
        "qué documentos", "que documentos",
        "qué archivos", "que archivos",
        "qué autores", "que autores",
        "tienes autores", "maneja el corpus",
        "cuántos textos", "cuantos textos",
        "de qué trata", "de que trata",
        "qué contiene", "que contiene",
         "qué instituciones", "que instituciones",
        "cuáles instituciones", "cuales instituciones",
        "qué dependencias", "que dependencias",
        "cuáles dependencias", "cuales dependencias",
        "qué organizaciones", "que organizaciones",
        "instituciones aparecen", "dependencias aparecen",
        "instituciones mencionan", "dependencias mencionan",
        "instituciones hay", "dependencias hay",
    ]
    q = question.lower()
    return any(k in q for k in keywords)

# ─── Extractor de autores ─────────────────────────────────────────────────────
def extraer_autores(chunks: List[Chunk]) -> dict:
    """Extrae autores, instituciones y dependencias del corpus."""
    autores = set()
    instituciones = set()

    # Patrones para autores personales
    patrones_autor = [
        re.compile(r'(?:autor(?:es)?|escrito por|elaborado por|coordinado por|por)[:\s]+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})', re.IGNORECASE),
        re.compile(r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+,\s[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)(?:\s*\(\d{4}\))', re.IGNORECASE),  # Apellido, Nombre (año)
    ]

    # Patrones para instituciones y dependencias
    patrones_inst = [
        re.compile(r'\b(INEGI|SESNSP|CNDH|SSP|FGR|PGR|SEDENA|SEMAR|INM|SAT|SEP|IMSS|ISSSTE|CONEVAL|CONAPO)\b'),
        re.compile(r'\b(Secretar[ií]a de[^,\n]{3,50}?)\b(?=,|\.|;|\n)', re.IGNORECASE),
        re.compile(r'\b(Comisi[oó]n[^,\n]{3,50}?)\b(?=,|\.|;|\n)', re.IGNORECASE),
        re.compile(r'\b(Instituto[^,\n]{3,50}?)\b(?=,|\.|;|\n)', re.IGNORECASE),
        re.compile(r'\b(Centro[^,\n]{3,50}?)\b(?=,|\.|;|\n)', re.IGNORECASE),
        re.compile(r'\b(M[eé]xico Eval[uú]a|Integralia|CIDAC|Animal Político|Nexos|Lantia)\b', re.IGNORECASE),
        re.compile(r'\b(ONU|UNODC|BID|Banco Mundial|OEA|OCDE)\b'),
    ]

    for ch in chunks:
        for patron in patrones_autor:
            for match in patron.findall(ch.text):
                nombre = match.strip()
                if len(nombre) > 5:
                    autores.add(nombre)

        for patron in patrones_inst:
            for match in patron.findall(ch.text):
                inst = match.strip()
                if len(inst) > 3:
                    instituciones.add(inst)

    return {
        "autores": sorted(autores),
        "instituciones": sorted(instituciones)
    }

# ─── Prompt para meta-preguntas ───────────────────────────────────────────────
def build_meta_prompt(question: str, chunks: List[Chunk]) -> str:
    docs_unicos = sorted({ch.doc_id for ch in chunks})
    resultado = extraer_autores(chunks)
    autores = resultado["autores"]
    instituciones = resultado["instituciones"]

    autores_str = (
        "\n".join(f"  - {a}" for a in autores)
        if autores else "  No se detectaron autores con nombre explícito."
    )

    inst_str = (
        "\n".join(f"  - {i}" for i in instituciones)
        if instituciones else "  No se detectaron instituciones."
    )

    contexto_meta = f"""Archivos en el corpus ({len(docs_unicos)}):
{chr(10).join(f'  - {d}' for d in docs_unicos)}

Total de archivos (corpus): {len(docs_unicos)}

Autores detectados ({len(autores)}):
{autores_str}

Instituciones y dependencias detectadas ({len(instituciones)}):
{inst_str}
"""

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Eres un asistente que responde preguntas sobre el corpus.
REGLA CRÍTICA: Responde ÚNICAMENTE con los datos que aparecen en INFORMACIÓN DEL CORPUS.
Si un dato no aparece, di exactamente: "El corpus no contiene esa información."
NUNCA inventes números, nombres ni afirmaciones que no estén en INFORMACIÓN DEL CORPUS.
<|eot_id|><|start_header_id|>user<|end_header_id|>

INFORMACIÓN DEL CORPUS:
{contexto_meta}

PREGUNTA:
{question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Basándome únicamente en la información del corpus:
"""

# ─── Prompt para preguntas sobre autores con contexto RAG ─────────────────────
def es_pregunta_sobre_autor(question: str) -> bool:
    keywords = [
        "qué dice", "que dice", "qué menciona", "que menciona",
        "qué opina", "que opina", "según el autor", "segun el autor",
        "qué plantea", "que plantea", "qué argumenta", "que argumenta",
    ]
    q = question.lower()
    return any(k in q for k in keywords)

# ─── Prompt RAG estándar ──────────────────────────────────────────────────────
def build_prompt(question: str, retrieved: List[Chunk]) -> str:
    sources = []
    for i, ch in enumerate(retrieved, start=1):
        sources.append(f"[{i}] [Documento: {ch.doc_id}]\n{ch.text}\n")
    sources_block = "\n".join(sources)
    doc_names = "\n".join(f"- {ch.doc_id}" for ch in retrieved)

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Eres un tutor analítico especializado en seguridad pública mexicana.
Responde usando únicamente el contexto proporcionado.
Al terminar tu respuesta, escribe siempre una sección "Fuentes:" listando los documentos consultados.
<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXTO RECUPERADO:
{sources_block}

PREGUNTA:
{question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# ─── Utilidades de documentos ─────────────────────────────────────────────────
def iter_docs(docs_dir: Path) -> Iterable[Path]:
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}:
            yield p

def read_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = ""
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
        except Exception as e:
            print(f"Advertencia: No se pudo leer el PDF {path.name}: {e}", file=sys.stderr)
        return text
    else:
        return path.read_text(encoding="utf-8", errors="replace")

def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def smart_chunk_text(text: str, chunk_chars: int = 1500, overlap_chars: int = 250) -> List[Tuple[str, int]]:
    text = normalize_whitespace(text)
    if not text:
        return []
    out: List[Tuple[str, int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        if end < n:
            last_period = text.rfind(".", start, end)
            last_newline = text.rfind("\n", start, end)
            cut_point = max(last_period, last_newline)
            if cut_point > start + (chunk_chars // 2):
                end = cut_point + 1
        chunk = text[start:end].strip()
        if chunk:
            out.append((chunk, start))
        if end >= n:
            break
        start = max(0, end - overlap_chars)
    return out

def ollama_embed(texts: List[str], model: str, base_url: str) -> np.ndarray:
    vecs = []
    for t in tqdm(texts, desc="Calculando Embeddings", unit="chunk"):
        r = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=120,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Embeddings failed ({r.status_code}): {r.text}")
        data = r.json()
        vecs.append(np.array(data["embedding"], dtype=np.float32))
    return np.vstack(vecs)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="RAG con modelo fine-tuneado LoRA local.")
    ap.add_argument("--docs",          required=True,  help="Directorio con documentos")
    ap.add_argument("--question",      required=True,  help="Pregunta del usuario")
    ap.add_argument("--lora-path",     required=True,  help="Ruta a la carpeta con los pesos LoRA")
    ap.add_argument("--base-model",    default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--topk",          type=int, default=4)
    ap.add_argument("--chunk-chars",   type=int, default=1500)
    ap.add_argument("--overlap-chars", type=int, default=250)
    ap.add_argument("--embed-model",   default="nomic-embed-text")
    ap.add_argument("--ollama-url",    default="http://localhost:11434")
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    # Cargar modelo fine-tuneado
    pipe = load_finetuned_model(args.lora_path, args.base_model)

    docs_dir = Path(args.docs).expanduser().resolve()
    if not docs_dir.exists():
        print(f"ERROR: no existe el directorio: {docs_dir}", file=sys.stderr)
        return 2

    index_path    = docs_dir / "vector_index.faiss"
    metadata_path = docs_dir / "chunks_metadata.pkl"
    chunks: List[Chunk] = []

    if index_path.exists() and metadata_path.exists() and not args.force_rebuild:
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            chunks = pickle.load(f)
    else:
        print("Construyendo nuevo índice FAISS...")
        files = sorted(iter_docs(docs_dir))
        if not files:
            print("ERROR: no se encontraron archivos.", file=sys.stderr)
            return 2
        for p in files:
            text = read_document(p)
            for chunk_text, start_char in smart_chunk_text(text, args.chunk_chars, args.overlap_chars):
                chunks.append(Chunk(doc_id=p.name, path=str(p.relative_to(docs_dir)), text=chunk_text, start_char=start_char))
        if not chunks:
            print("ERROR: no se generaron chunks.", file=sys.stderr)
            return 2
        X = ollama_embed([c.text for c in chunks], model=args.embed_model, base_url=args.ollama_url)
        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        faiss.write_index(index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(chunks, f)
        print("Índice guardado correctamente.")

    # ─── Enrutador de preguntas ───────────────────────────────────────────────
    if es_meta_pregunta(args.question):
        # Meta-pregunta: sobre estructura del corpus
        prompt = build_meta_prompt(args.question, chunks)
        answer = local_generate(prompt, pipe)
    else:
        # Pregunta normal: búsqueda RAG
        print("\nCalculando embedding de la pregunta...")
        q_vec = ollama_embed([args.question], model=args.embed_model, base_url=args.ollama_url)[0:1]
        faiss.normalize_L2(q_vec)
        scores, idxs = index.search(q_vec, k=min(args.topk, len(chunks)))
        retrieved = [chunks[i] for i in idxs[0].tolist()]

        print("\n=== Contexto Recuperado ===")
        for rank, (ch, score) in enumerate(zip(retrieved, scores[0].tolist()), start=1):
            snippet = ch.text.replace('\n', ' ')[:200]
            print(f"[{rank}] Score: {score:.4f} | Archivo: {ch.doc_id}")
            print(f"     Extracto: {snippet}...\n")

        print("Generando respuesta con modelo fine-tuneado...")
        prompt = build_prompt(args.question, retrieved)
        answer = local_generate(prompt, pipe)

    print("\n=== RESPUESTA ===")
    print(answer)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())