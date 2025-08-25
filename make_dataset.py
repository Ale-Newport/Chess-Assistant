# make_dataset.py
import os, random, math
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import cv2

# === Config ===
OUT_DIR = Path("dataset")
PIECES_DIR = Path("piezas")  # wP.png, wN.png, ... con fondo transparente
IMG_SIZE = (150, 150)        # tamaño final de cada casilla (ancho, alto)
SAMPLES_PER_CLASS = 400      # cuántas imágenes generar por clase en total (suma de todos los temas/zooms)
RANDOM_SEED = 1337
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

LABELS = ["wP","wN","wB","wR","wQ","wK","bP","bN","bB","bR","bQ","bK","empty"]

# Paletas aproximadas de chess.com (light, dark) en BGR
THEMES = {
    "classic": ((211,236,235), (90,148,122)),     # #ebecd3 / #7a945a
    "tournament": ((230,226,214), (121,146,98)),  # aprox
    "blue": ((232,240,247), (120,155,191)),
    "bubblegum": ((245,231,222), (213,114,118)),
    "wood": ((222,208,186), (153,117,80)),
}

# Factores de “zoom” (escala de la pieza respecto a la casilla)
ZOOMS = [0.72, 0.78, 0.84, 0.90, 0.96, 1.02]

# Augmentations
ROT_MAX_DEG = 4            # rotación aleatoria ±
BRIGHT_JITTER = 0.10       # +/- 10%
ALPHA_JITTER = 0.05        # +/- 5% en opacidad de la pieza
BLUR_PROB = 0.25
BLUR_KS = [3,5]            # kernel blur
NOISE_PROB = 0.20          # añadir ruido gaussiano

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lbl in LABELS:
        (OUT_DIR / lbl).mkdir(parents=True, exist_ok=True)

def load_pieces() -> Dict[str, np.ndarray]:
    pieces = {}
    for lbl in LABELS:
        if lbl == "empty": 
            continue
        path = PIECES_DIR / f"{lbl}.png"
        if not path.exists():
            print(f"⚠️ Falta {path} — esa clase se generará solo como vacía si aplica.")
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # RGBA esperado
        if img is None:
            print(f"⚠️ No pude leer {path}")
            continue
        pieces[lbl] = img
    return pieces

def draw_square(light_bgr: Tuple[int,int,int], dark_bgr: Tuple[int,int,int], light=True) -> np.ndarray:
    bg = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), np.uint8)
    color = light_bgr if light else dark_bgr
    bg[:] = color
    return bg

def rand_bool(p): 
    return random.random() < p

def jitter_brightness(bgr: np.ndarray, jitter: float) -> np.ndarray:
    factor = 1.0 + random.uniform(-jitter, jitter)
    out = np.clip(bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out

def add_noise(bgr: np.ndarray) -> np.ndarray:
    sigma = random.uniform(4, 10)
    noise = np.random.normal(0, sigma, bgr.shape).astype(np.float32)
    out = np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

def rotate_rgba(img_rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rota manteniendo tamaño (rellena con alpha=0)."""
    h, w = img_rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(img_rgba, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))

def composite_piece_on_square(bg_bgr: np.ndarray, piece_rgba: np.ndarray,
                              scale: float, alpha_boost: float = 0.0) -> np.ndarray:
    """Escala/rota pieza y la pega centrada; recorta si se sale para evitar broadcasting."""
    canvas = bg_bgr.copy()
    H, W = canvas.shape[:2]

    # -- escalar manteniendo aspecto por ancho
    target_w = int(W * scale)
    target_w = max(1, min(target_w, W * 2))  # limita por si el zoom es grande
    target_h = int(piece_rgba.shape[0] * (target_w / piece_rgba.shape[1]))
    target_h = max(1, min(target_h, H * 2))

    piece = cv2.resize(piece_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # -- rotación leve (manteniendo tamaño de la pieza actual)
    if ROT_MAX_DEG > 0:
        angle = random.uniform(-ROT_MAX_DEG, ROT_MAX_DEG)
        piece = rotate_rgba(piece, angle)

    ph, pw = piece.shape[:2]
    # Colocar centrado
    x = (W - pw) // 2
    y = (H - ph) // 2

    # -- recortar a los bordes de la casilla
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + pw); y1 = min(H, y + ph)
    if x0 >= x1 or y0 >= y1:
        # La pieza quedó totalmente fuera (raro, pero por seguridad)
        return canvas

    # recorte correspondiente en la pieza
    px0 = x0 - x
    py0 = y0 - y
    px1 = px0 + (x1 - x0)
    py1 = py0 + (y1 - y0)

    sub = piece[py0:py1, px0:px1]
    if sub.shape[2] == 4:
        alpha = sub[:, :, 3].astype(np.float32) / 255.0
        if alpha_boost != 0.0:
            alpha = np.clip(alpha * (1.0 + alpha_boost), 0.0, 1.0)
        for c in range(3):
            canvas[y0:y1, x0:x1, c] = (1 - alpha) * canvas[y0:y1, x0:x1, c] + alpha * sub[:, :, c]
    else:
        canvas[y0:y1, x0:x1] = sub[:, :, :3]

    return canvas

def maybe_postprocess(bgr: np.ndarray) -> np.ndarray:
    out = jitter_brightness(bgr, BRIGHT_JITTER)
    if rand_bool(NOISE_PROB):
        out = add_noise(out)
    if rand_bool(BLUR_PROB):
        k = random.choice(BLUR_KS)
        out = cv2.GaussianBlur(out, (k,k), 0)
    return out

def generate_dataset():
    ensure_dirs()
    pieces = load_pieces()

    # Repartir muestras por tema y zoom
    combos: List[Tuple[str,float]] = [(t, z) for t in THEMES.keys() for z in ZOOMS]
    per_combo = max(1, SAMPLES_PER_CLASS // max(1, len(combos)))

    for label in LABELS:
        count = 0
        for theme_name, zoom in combos:
            light, dark = THEMES[theme_name]
            for _ in range(per_combo):
                light_square = ((count + _) % 2 == 0)  # alterna claro/oscuro
                bg = draw_square(light, dark, light_square)

                if label == "empty" or label not in pieces:
                    img = maybe_postprocess(bg)
                else:
                    piece = pieces[label]
                    a_boost = random.uniform(-ALPHA_JITTER, ALPHA_JITTER)
                    comp = composite_piece_on_square(bg, piece, zoom, alpha_boost=a_boost)
                    img = maybe_postprocess(comp)

                out_path = OUT_DIR / label / f"{label}_{theme_name}_z{str(zoom).replace('.','')}_{count:05d}.png"
                cv2.imwrite(str(out_path), img)
                count += 1

        print(f"✅ {label}: {count} imágenes")

if __name__ == "__main__":
    print("🎯 Generando dataset sintético estilo chess.com…")
    generate_dataset()
    print("📦 Listo en ./dataset/")
