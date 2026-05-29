import os
import re
import urllib.request
import time
import json
from PIL import Image
from io import BytesIO

# ==========================================
# CONFIGURACIÓN DEL DATASET
# ==========================================
BASE_DIR       = 'datos'
IMG_SIZE       = (128, 128)
OBJETIVO       = 1000        # imágenes por clase
TAMANO_IMAGEN  = "medium"
AWS_S3_BASE    = "https://inaturalist-open-data.s3.amazonaws.com/photos/"

CLASES = {
    "ballena": 152904,   # Cetacea (todas las ballenas y delfines)
    "pajaro":  3,        # Aves (todos los pájaros)
    "arana":   47118,    # Araneae (todas las arañas)
    "mono":    43367,    # Primates (todos los monos)
}


def redimensionar(data_bytes, size):
    """Recibe los bytes de la imagen y devuelve una imagen PIL redimensionada."""
    img = Image.open(BytesIO(data_bytes))
    if img.width < 80 or img.height < 80:
        return None
    img = img.convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return img


def descargar_clase(nombre, taxon_id, objetivo, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  [{nombre.upper()}]  taxon_id={taxon_id}  objetivo={objetivo}")
    print(f"  Carpeta: {target_dir}")
    print(f"{'='*50}")

    descargadas = 0
    page        = 1

    while descargadas < objetivo:
        print(f"\n  --- Página {page} ---")

        url_api = (
            f"https://api.inaturalist.org/v1/observations"
            f"?taxon_id={taxon_id}"
            f"&only_id=false&per_page=200&page={page}&licensed=true"
        )

        try:
            req = urllib.request.Request(url_api, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            print(f"  Error al conectar (página {page}): {e}. Reintentando en 10s...")
            time.sleep(10)
            continue

        results = data.get('results', [])
        if not results:
            print("  No hay más imágenes disponibles en iNaturalist para esta clase.")
            break

        for obs in results:
            if descargadas >= objetivo:
                break
            for photo in obs.get('photos', []):
                if descargadas >= objetivo:
                    break

                photo_id  = photo.get('id')
                url_prev  = photo.get('url', '')

                match = re.search(r'\.(\w+)\?', url_prev)
                ext   = match.group(1) if match else 'jpg'

                if not photo_id:
                    continue

                filename = f"{nombre}_{photo_id}.jpg"   # siempre .jpg tras redimensionar
                filepath = os.path.join(target_dir, filename)

                if os.path.exists(filepath):
                    descargadas += 1
                    continue

                aws_url = f"{AWS_S3_BASE}{photo_id}/{TAMANO_IMAGEN}.{ext}"

                try:
                    with urllib.request.urlopen(aws_url) as r:
                        img_bytes = r.read()

                    img = redimensionar(img_bytes, IMG_SIZE)
                    if img is None:
                        continue

                    img.save(filepath, "JPEG", quality=90)
                    descargadas += 1
                    print(f"  [{descargadas}/{objetivo}] {filename}", end="\r")

                except Exception:
                    continue

        page += 1
        time.sleep(1.5)   # pausa de cortesía entre páginas

    print(f"\n  ✓ {descargadas} imágenes guardadas en {target_dir}/")
    return descargadas


# ==========================================
# DESCARGA DE TODAS LAS CLASES
# ==========================================
if __name__ == "__main__":
    print(f"Tamaño de imagen: {IMG_SIZE[0]}x{IMG_SIZE[1]} px")
    print(f"Objetivo por clase: {OBJETIVO}")
    print(f"Carpeta base: {os.path.abspath(BASE_DIR)}\n")

    resumen = {}
    for nombre, taxon_id in CLASES.items():
        target_dir = os.path.join(BASE_DIR, nombre)
        n = descargar_clase(nombre, taxon_id, OBJETIVO, target_dir)
        resumen[nombre] = n

    print("\n\n── Resumen final ─────────────────────────────")
    total = 0
    for nombre, n in resumen.items():
        barra = "█" * (n // 50)
        print(f"  {nombre:<12} {n:>5} imgs  {barra}")
        total += n
    print(f"  {'TOTAL':<12} {total:>5} imgs")
    print(f"\nDataset listo en: {os.path.abspath(BASE_DIR)}/")
    print("\nEstructura:")
    print("  datos/")
    for nombre in CLASES:
        print(f"    {nombre}/   *.jpg")