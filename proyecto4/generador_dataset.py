import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

# Necesitamos re-declarar la clase para que pickle la reconozca al cargar el archivo
@dataclass(frozen=True)
class Chunk:
    doc_id: str
    path: str
    text: str
    start_char: int

def generate_qa_pair(chunk: Chunk, base_url="http://localhost:11434", model="llama3.2:3b") -> dict:
    """Pide a Ollama que genere un Q&A basado en el chunk, forzando la salida a JSON."""
    
    prompt = f"""Eres un experto creador de datasets de entrenamiento. Tu objetivo es crear UNA interacción entre un usuario y un Tutor Analítico basándote estrictamente en el texto proporcionado.

REGLAS PARA LA PREGUNTA DEL USUARIO:
1. Relevancia estricta: La pregunta DEBE tratar exclusivamente sobre el tema central, los datos, lugares o los eventos específicos que se mencionan en este fragmento exacto.
2. Diversidad obligatoria: PROHIBIDO hacer preguntas genéricas. NUNCA uses la frase "¿Cuál fue la tasa de incidencia delictiva?". Imagina que el usuario es un analista buscando un dato muy particular de este párrafo.

REGLAS PARA LA RESPUESTA DEL TUTOR:
1. Tono Socrático: El tutor no debe dar conclusiones cerradas, sino guiar con preguntas analíticas.
2. Tono Académico: Absolutamente neutral y objetivo.
3. Citas: Debe incluir obligatoriamente al final de su respuesta la referencia exacta: [Documento: {chunk.doc_id}]

TEXTO FUENTE:
{chunk.text}

Devuelve ÚNICAMENTE un objeto JSON válido con las claves "user" (la pregunta sobre el texto) y "assistant" (la respuesta del tutor). No incluyas texto markdown, solo el JSON raw.
Ejemplo:
{{
  "user": "¿Cuál fue la tasa de incidencia delictiva?",
  "assistant": "Para analizar este punto, observemos los datos. El reporte indica una tasa de 15%. ¿Qué factores socioeconómicos crees que influyeron en esta métrica? [Documento: {chunk.doc_id}]"
}}
"""
    try:
        r = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json", # Ollama fuerza la salida a JSON
                "options": {"temperature": 0.3}
            },
            timeout=120,
        )
        if r.status_code == 200:
            response_text = r.json().get("response", "").strip()
            return json.loads(response_text)
    except Exception as e:
        print(f"Error generando con Ollama: {e}", file=sys.stderr)
    return None

def main():
    metadata_path = Path("docs/chunks_metadata.pkl") # Ajusta la ruta a tu carpeta
    output_jsonl = Path("dataset_entrenamiento.jsonl")
    
    if not metadata_path.exists():
        print(f"ERROR: No se encontró {metadata_path}. Corre el RAG primero.", file=sys.stderr)
        return

    print("Cargando fragmentos extraídos...")
    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)

    # Para empezar, tomaremos una muestra aleatoria de 50 chunks para probar. 
    # Cuando quieras tu dataset final de 1,000 entradas, cambia este valor.
    sample_size = min(50, len(chunks))
    sampled_chunks = random.sample(chunks, sample_size)

    print(f"Generando dataset con {sample_size} ejemplos usando Ollama...")
    
    system_prompt = "Eres un tutor analítico especializado en seguridad pública en México. Responde con neutralidad, usa el método socrático para análisis complejos y cita tus fuentes."
    
    success_count = 0
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for chunk in tqdm(sampled_chunks, desc="Generando Q&A"):
            qa_dict = generate_qa_pair(chunk)
            
            if qa_dict and "user" in qa_dict and "assistant" in qa_dict:
                # Estructuramos en formato ChatML (Messages) ideal para Fine-Tuning
                chatml_entry = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Basado en este contexto:\n{chunk.text}\n\n{qa_dict['user']}"},
                        {"role": "assistant", "content": qa_dict['assistant']}
                    ]
                }
                f_out.write(json.dumps(chatml_entry, ensure_ascii=False) + "\n")
                success_count += 1

    print(f"\n¡Éxito! Dataset guardado en {output_jsonl} con {success_count} ejemplos válidos.")

if __name__ == "__main__":
    main()