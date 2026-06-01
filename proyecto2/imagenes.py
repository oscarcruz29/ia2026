"""
Descarga imágenes de animales desde Open Images (Google) usando fiftyone.
No requiere cuenta ni API key.

Instalar:
    pip install fiftyone Pillow

Clases disponibles en Open Images:
    Whale, Bird, Spider, Monkey, Frog
"""

import os
from pathlib import Path
from PIL import Image

import fiftyone as fo
import fiftyone.zoo as foz

# ─── Configuración ────────────────────────────────────────────────────────────

IMG_SIZE   = (128, 128)   # Cambia a (224, 224) si usas transfer learning
MAX_IMGS   = 500          # Imágenes por clase
OUTPUT_DIR = Path("dataset")

# Nombres exactos de las clases en Open Images (mayúscula al inicio)
CLASES = {
    "ballena": "Whale",
    "pajaro":  "Bird",
    "arana":   "Spider",
    "mono":    "Monkey",
    "rana":    "Frog",    # Clase agregada para las ranas
}

# ─── Descarga desde Open Images ───────────────────────────────────────────────

def descargar_clase(nombre_local, nombre_openimages, max_imgs):
    print(f"\n[{nombre_local.upper()}] Descargando {max_imgs} imágenes de Open Images...")

    # fiftyone descargará la metadata del dataset la primera vez que se ejecute
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",               # "train" tiene la mayor cantidad de imágenes
        label_types=["detections"],  # filtra imágenes donde aparece el animal
        classes=[nombre_openimages],
        max_samples=max_imgs,
        dataset_name=f"oi_{nombre_local}",
        overwrite=True,
    )

    carpeta_dest = OUTPUT_DIR / nombre_local
    carpeta_dest.mkdir(parents=True, exist_ok=True)

    guardadas = 0
    for sample in dataset:
        src = sample.filepath
        if not os.path.exists(src):
            continue
        try:
            with Image.open(src) as img:
                # Descartar miniaturas o íconos
                if img.width < 80 or img.height < 80:
                    continue
                img = img.convert("RGB")
                
                # Soporte compatible para versiones antiguas y nuevas de Pillow
                resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                img = img.resize(IMG_SIZE, resample_filter)
                
                dst = carpeta_dest / f"{nombre_local}_{guardadas:04d}.jpg"
                img.save(dst, "JPEG", quality=90)
                guardadas += 1
        except Exception:
            continue

    # Limpiar dataset interno de fiftyone para liberar espacio en disco
    fo.delete_dataset(f"oi_{nombre_local}")

    print(f"  ✓ {guardadas} imágenes guardadas en {carpeta_dest}/")
    return guardadas


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Tamaño: {IMG_SIZE[0]}x{IMG_SIZE[1]} px  |  Objetivo: {MAX_IMGS} imgs/clase\n")

    resumen = {}
    for nombre_local, nombre_oi in CLASES.items():
        n = descargar_clase(nombre_local, nombre_oi, MAX_IMGS)
        resumen[nombre_local] = n

    print("\n── Resumen final ────────────────────────")
    total = 0
    for animal, n in resumen.items():
        barra = "█" * (n // 20)
        print(f"  {animal:<12} {n:>4} imgs  {barra}")
        total += n
    print(f"  {'TOTAL':<12} {total:>4} imgs")
    print(f"\nDataset listo en: {OUTPUT_DIR.resolve()}/")

if __name__ == "__main__":
    main()