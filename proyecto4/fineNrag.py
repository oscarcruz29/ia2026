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

# ─── Carga del modelo fine-tuneado (se hace UNA sola vez al arrancar) ─────────
def load_finetuned_model(lora_path: str, base_model_id: str = "unsloth/Llama-3.2-3B-Instruct"):
    print(f"Cargando modelo base: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Cargando adaptadores LoRA desde: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
    )
    print("✅ Modelo fine-tuneado listo.")
    return pipe

# ─── Generación con tu modelo local (reemplaza ollama_generate) ───────────────
def local_generate(prompt: str, pipe) -> str:
    output = pipe(prompt)
    # Extraemos solo el texto generado después del prompt
    full_text = output[0]["generated_text"]
    # Si el modelo repite el prompt, cortamos solo la respuesta
    if prompt in full_text:
        return full_text[len(prompt):].strip()
    return full_text.strip()

# ─── Embeddings siguen usando Ollama (nomic-embed-text) ───────────────────────
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

def build_prompt(question: str, retrieved: List[Chunk]) -> str:
    sources = []
    for i, ch in enumerate(retrieved, start=1):
        sources.append(f"[Documento: {ch.doc_id}]\n{ch.text}\n")
    sources_block = "\n".join(sources)

    return f"""Eres un tutor analítico especializado en seguridad pública. Responde de forma precisa y rigurosa.

Instrucciones:
- Responde usando ÚNICAMENTE la información contenida en el CONTEXTO.
- Si el CONTEXTO no contiene suficiente información, responde: "La información proporcionada en el corpus no detalla este aspecto."
- Cita SIEMPRE el documento al final de tu afirmación usando el formato: [Documento: NombreDelArchivo.pdf].

CONTEXTO RECUPERADO:
{sources_block}

PREGUNTA DEL USUARIO:
{question}

RESPUESTA:
"""

def main() -> int:
    ap = argparse.ArgumentParser(description="RAG con modelo fine-tuneado LoRA local.")
    ap.add_argument("--docs",        required=True,  help="Directorio con documentos")
    ap.add_argument("--question",    required=True,  help="Pregunta del usuario")
    ap.add_argument("--lora-path",   required=True,  help="Ruta a la carpeta con los pesos LoRA")
    ap.add_argument("--base-model",  default="unsloth/Llama-3.2-3B-Instruct", help="Modelo base HuggingFace")
    ap.add_argument("--topk",        type=int, default=4)
    ap.add_argument("--chunk-chars", type=int, default=1500)
    ap.add_argument("--overlap-chars", type=int, default=250)
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--ollama-url",  default="http://localhost:11434")
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    # Cargar modelo fine-tuneado al inicio
    pipe = load_finetuned_model(args.lora_path, args.base_model)

    docs_dir = Path(args.docs).expanduser().resolve()
    if not docs_dir.exists():
        print(f"ERROR: no existe el directorio: {docs_dir}", file=sys.stderr)
        return 2

    index_path    = docs_dir / "vector_index.faiss"
    metadata_path = docs_dir / "chunks_metadata.pkl"
    chunks: List[Chunk] = []

    if index_path.exists() and metadata_path.exists() and not args.force_rebuild:
        print("Cargando índice FAISS y metadatos desde el disco...")
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
    prompt  = build_prompt(args.question, retrieved)
    answer  = local_generate(prompt, pipe)

    print("\n=== RESPUESTA ===")
    print(answer)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())