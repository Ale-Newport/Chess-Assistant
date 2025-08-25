# capture_board_crop.py (con refinamiento)
import time, sys
import numpy as np, cv2, tensorflow as tf
from PIL import ImageGrab

MODEL_PATH = "board_2p.h5"
INPUT = 384
WAIT_SECONDS = 3
OUT_CROP = "board_crop.png"
OUT_GRID = "board_grid.png"

def letterbox(img, size=INPUT):
    h,w = img.shape[:2]
    s = min(size/h, size/w)
    nh, nw = int(h*s), int(w*s)
    rsz = cv2.resize(img,(nw,nh), cv2.INTER_AREA)
    canvas = np.zeros((size,size,3), np.uint8)
    top,left = (size-nh)//2, (size-nw)//2
    canvas[top:top+nh, left:left+nw] = rsz
    return canvas, s, left, top

def predict_tl_br(model, frame):
    lb, s, left, top = letterbox(frame, INPUT)
    x = (lb.astype(np.float32)/255.0)[None,...]
    tlx,tly,brx,bry = model.predict(x, verbose=0)[0]*INPUT
    tl = np.array([(tlx-left)/s, (tly-top)/s], np.float32)
    br = np.array([(brx-left)/s, (bry-top)/s], np.float32)
    return tl, br

def checkerboard_score(img, tl, br):
    x1,y1 = tl.astype(int); x2,y2 = br.astype(int)
    if x2<=x1 or y2<=y1: return -1e9
    crop = img[max(0,y1):min(img.shape[0],y2), max(0,x1):min(img.shape[1],x2)]
    if crop.size == 0: return -1e9
    crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (256,256))
    # divide 8x8, medias por paridad
    s = 32
    even, odd = [], []
    for r in range(8):
        for c in range(8):
            cell = crop[r*s:(r+1)*s, c*s:(c+1)*s]
            (even if (r+c)%2==0 else odd).append(cell.mean())
    even, odd = np.mean(even), np.mean(odd)
    return abs(even-odd)

def refine_box(img, tl, br, steps=7, frac=0.03):
    """Pequeña búsqueda local alrededor de TL/BR."""
    h,w = img.shape[:2]
    dx = int(frac*(br[0]-tl[0])); dy = int(frac*(br[1]-tl[1]))
    best = (tl.copy(), br.copy()); best_s = checkerboard_score(img, tl, br)
    for i in range(-steps, steps+1):
        for j in range(-steps, steps+1):
            for k in range(-steps, steps+1):
                for m in range(-steps, steps+1):
                    tl2 = np.array([tl[0]+i*dx/steps, tl[1]+j*dy/steps])
                    br2 = np.array([br[0]+k*dx/steps, br[1]+m*dy/steps])
                    # fuerza cuadrado (anchura=altura)
                    side = min(br2[0]-tl2[0], br2[1]-tl2[1])
                    if side<=20: continue
                    br_sq = np.array([tl2[0]+side, tl2[1]+side])
                    score = checkerboard_score(img, tl2, br_sq)
                    if score > best_s:
                        best_s, best = score, (tl2, br_sq)
    return best

def main():
    print(f"⏳ Capturando pantalla en {WAIT_SECONDS}s…")
    time.sleep(WAIT_SECONDS)

    shot = ImageGrab.grab()
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)

    print("🔍 Cargando modelo…")
    model = tf.keras.models.load_model(MODEL_PATH)

    tl, br = predict_tl_br(model, frame)
    tl, br = refine_box(frame, tl, br)  # <<< refinamiento
    x1,y1 = tl.astype(int); x2,y2 = br.astype(int)

    x1 = max(0,min(frame.shape[1]-1,x1)); x2 = max(0,min(frame.shape[1],x2))
    y1 = max(0,min(frame.shape[0]-1,y1)); y2 = max(0,min(frame.shape[0],y2))
    if x2<=x1 or y2<=y1:
        print("❌ No se obtuvo rectángulo válido"); return

    crop = frame[y1:y2, x1:x2].copy()
    cv2.imwrite(OUT_CROP, crop)
    vis = frame.copy()
    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),3)
    cv2.imwrite(OUT_GRID, vis)
    print("✅ Guardados:", OUT_CROP, OUT_GRID)

if __name__ == "__main__":
    main()
