# make_synth_boards.py
import os, json, random
from pathlib import Path
import numpy as np, cv2
from PIL import ImageGrab, Image

OUT_DIR = Path("data/images")
LBL_PATH = Path("data/labels.jsonl")
N = 1200                        # cuántas imágenes quieres
CANVAS_SIZES = [(1440,900),(1280,800),(1728,1117),(1680,1050)]
THEMES = [                      # BGR aprox de varios temas chess.com
    ((211,236,235),(90,148,122)),     # green clásico
    ((235,227,202),(132,82,72)),      # brown
    ((235,227,202),(120,100,160)),    # purple
    ((235,227,202),(116,145,170)),    # blue
]
SHADOW = True                   # sombra sutil al tablero

def checkerboard(tile_light, tile_dark, size_px):
    board = np.zeros((size_px,size_px,3), np.uint8)
    sq = size_px // 8
    for y in range(8):
        for x in range(8):
            color = tile_light if (x+y)%2==0 else tile_dark
            cv2.rectangle(board,(x*sq,y*sq),((x+1)*sq,(y+1)*sq),color,-1)
    return board

def get_background(w,h):
    # intenta usar pantallazo como fondo (más realista)
    try:
        shot = ImageGrab.grab()
        bg = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
        # recorte aleatorio del tamaño del canvas
        if bg.shape[1] >= w and bg.shape[0] >= h:
            x = random.randint(0, bg.shape[1]-w)
            y = random.randint(0, bg.shape[0]-h)
            return bg[y:y+h, x:x+w].copy()
    except Exception:
        pass
    # degradado/ruido de respaldo
    a = np.random.randint(40,80)
    b = np.random.randint(12,24)
    grad = np.linspace(a,a+b,h, dtype=np.uint8)[:,None]
    bg = np.dstack([grad+np.random.randint(0,8,(h,w),np.uint8) for _ in range(3)])
    return bg

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LBL_PATH, "w") as f:
        for i in range(N):
            w,h = random.choice(CANVAS_SIZES)
            canvas = get_background(w,h)

            light,dark = random.choice(THEMES)
            size = random.randint(int(0.45*min(w,h)), int(0.85*min(w,h)))
            board = checkerboard(light,dark,size)

            # pequeñas variaciones de color/contraste
            alpha = 0.85 + 0.3*random.random()
            board = np.clip(board.astype(np.float32)*alpha,0,255).astype(np.uint8)
            if random.random()<0.3:
                board = cv2.GaussianBlur(board,(3,3),0)

            # posición (deja márgenes)
            mx = int(0.03*w); my = int(0.03*h)
            x1 = random.randint(mx, max(mx, w-size-mx))
            y1 = random.randint(my, max(my, h-size-my))
            x2, y2 = x1+size, y1+size

            if SHADOW:
                s = 12
                shadow = cv2.GaussianBlur(np.ones((size,size,3),np.uint8)*0,(21,21),0)
                canvas[y1+s:y2+s, x1+s:x2+s] = cv2.addWeighted(
                    canvas[y1+s:y2+s, x1+s:x2+s], 1.0, shadow, 0.25, 0
                )

            canvas[y1:y2, x1:x2] = board

            # “UI overlays” simples (simulan tooltips/botones)
            if random.random()<0.35:
                rx1 = random.randint(0, w-220); ry1 = random.randint(0, h-120)
                cv2.rectangle(canvas, (rx1,ry1), (rx1+220, ry1+80), (40,40,40), -1)
                cv2.putText(canvas, "hint", (rx1+14,ry1+48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,220,220), 2)

            name = f"synth_{i:05d}.png"
            cv2.imwrite(str(OUT_DIR/name), canvas)
            f.write(json.dumps({"file": name, "tl":[x1,y1], "br":[x2,y2]})+"\n")

    print(f"✅ Generadas {N} imágenes en {OUT_DIR}")
    print(f"📝 Etiquetas: {LBL_PATH}")

if __name__ == "__main__":
    main()
