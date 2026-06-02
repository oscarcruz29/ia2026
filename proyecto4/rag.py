import argparse
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF para procesar PDFs
import numpy as np
import requests
from tqdm import tqdm
import faiss

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    path: str
    text: str
    start_char: int

def iter_docs(docs_dir: Path) -> Iterable[Path]:
    """Busca archivos .txt, .md y .pdf en el directorio."""
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}:
            yield p

def read_document(path: Path) -> str:
    """Extrae texto dependiendo de la extensión del archivo."""
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
    """
    Chunking que intenta cortar en el punto final más cercano para no romper oraciones,
    vital para mantener el sentido semántico en reportes de seguridad.
    """
    text = normalize_whitespace(text)
    if not text:
        return []
    
    out: List[Tuple[str, int]] = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(n, start + chunk_chars)
        
        # Si no estamos al final del texto, intentamos retroceder hasta el último punto o salto de línea
        if end < n:
            last_period = text.rfind(".", start, end)
            last_newline = text.rfind("\n", start, end)
            
            # Preferimos cortar en un punto o salto de línea si está en la última mitad del chunk
            cut_point = max(last_period, last_newline)
            if cut_point > start + (chunk_chars // 2):
                end = cut_point + 1 # Incluir el punto o salto de línea
                
        chunk = text[start:end].strip()
        if chunk:
            out.append((chunk, start))
            
        if end >= n:
            break
            
        # Calcular el inicio del siguiente chunk basado en el overlap
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

def ollama_generate(prompt: str, model: str, base_url: str, temperature: float = 0.2) -> str:
    r = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Generate failed ({r.status_code}): {r.text}")
    return r.json().get("response", "").strip()

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
    ap = argparse.ArgumentParser(description="RAG Híbrido con soporte PDF, Chunking Semántico y Persistencia.")
    ap.add_argument("--docs", required=True, help="Directorio con documentos (PDF, TXT, MD)")
    ap.add_argument("--question", required=True, help="Pregunta del usuario")
    ap.add_argument("--topk", type=int, default=4, help="Número de chunks a recuperar")
    ap.add_argument("--chunk-chars", type=int, default=1500, help="Tamaño de chunk (caracteres)")
    ap.add_argument("--overlap-chars", type=int, default=250, help="Solapamiento (caracteres)")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Modelo de embeddings en Ollama")
    ap.add_argument("--llm-model", default="llama3.2:3b", help="Modelo generativo en Ollama")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="URL base de Ollama")
    ap.add_argument("--force-rebuild", action="store_true", help="Fuerza la reconstrucción del índice vectorial")
    args = ap.parse_args()

    docs_dir = Path(args.docs).expanduser().resolve()
    if not docs_dir.exists():
        print(f"ERROR: no existe el directorio: {docs_dir}", file=sys.stderr)
        return 2

    # Rutas para la persistencia del índice vectorial y los metadatos
    index_path = docs_dir / "vector_index.faiss"
    metadata_path = docs_dir / "chunks_metadata.pkl"

    chunks: List[Chunk] = []
    
    # Lógica de Persistencia (Carga o Construcción)
    if index_path.exists() and metadata_path.exists() and not args.force_rebuild:
        print("Cargando índice FAISS y metadatos desde el disco...")
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            chunks = pickle.load(f)
    else:
        print("Construyendo nuevo índice FAISS a partir de los documentos...")
        files = sorted(iter_docs(docs_dir))
        if not files:
            print("ERROR: no se encontraron archivos PDF, TXT o MD.", file=sys.stderr)
            return 2

        for p in files:
            text = read_document(p)
            for chunk_text, start_char in smart_chunk_text(text, chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars):
                chunks.append(
                    Chunk(
                        doc_id=p.name,
                        path=str(p.relative_to(docs_dir)),
                        text=chunk_text,
                        start_char=start_char,
                    )
                )

        if not chunks:
            print("ERROR: no se generaron chunks (documentos vacíos o ilegibles).", file=sys.stderr)
            return 2

        chunk_texts = [c.text for c in chunks]
        X = ollama_embed(chunk_texts, model=args.embed_model, base_url=args.ollama_url)

        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        
        # Guardar en disco para futuras consultas
        faiss.write_index(index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(chunks, f)
        print("Índice y metadatos guardados correctamente.")

    # Flujo de Búsqueda (Inferencia)
    print("\nCalculando embedding de la pregunta...")
    q_vec = ollama_embed([args.question], model=args.embed_model, base_url=args.ollama_url)[0:1]
    faiss.normalize_L2(q_vec)

    scores, idxs = index.search(q_vec, k=min(args.topk, len(chunks)))
    retrieved = [chunks[i] for i in idxs[0].tolist()]

    print("\n=== Contexto Recuperado (RAG) ===")
    for rank, (ch, score) in enumerate(zip(retrieved, scores[0].tolist()), start=1):
        snippet = ch.text.replace('\n', ' ')
        snippet = snippet if len(snippet) <= 200 else (snippet[:200] + " ...")
        print(f"[{rank}] Score: {score:.4f} | Archivo: {ch.doc_id}")
        print(f"     Extracto: {snippet}\n")

    print("Generando respuesta del Tutor Analítico...")
    prompt = build_prompt(args.question, retrieved)
    answer = ollama_generate(prompt, model=args.llm_model, base_url=args.ollama_url, temperature=0.2)

    print("\n=== RESPUESTA ===")
    print(answer)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())