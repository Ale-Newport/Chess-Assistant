# tablero.py
import os, sys, time, signal, math, itertools, random
import numpy as np
import cv2
from PIL import ImageGrab
import tensorflow as tf

# ======== CONFIG ========
TIEMPO_ESPERA = 2
JUEGO_CON_BLANCAS = True
CASILLA_SIZE = (64, 64)               # input del modelo
MODELO_PATH = "modelo_piezas.h5"
CLASSES = ["wP","wN","wB","wR","wQ","wK","bP","bN","bB","bR","bQ","bK","empty"]

# Detección tablero
MIN_BOARD_SIZE = 420
HOUGH_TRIES = [
    dict(canny=(50,150), dilate=0,   thr=120, minFrac=0.22, gap=8),
    dict(canny=(30,120), dilate=1,   thr=110, minFrac=0.18, gap=10),
    dict(canny=(20,100), dilate=2,   thr=100, minFrac=0.15, gap=12),
]
UNIFORM_STD_TOL = 14
CLICK_TO_CROP_FALLBACK = True

# Visualización
SQ = 80
CLR_LIGHT = (240,217,181)
CLR_DARK  = (181,136,99)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ======== CTRL+C ========
def cleanup(*_):
    try: cv2.destroyAllWindows()
    except: pass
    print("\n👋 Cancelado.")
    sys.exit(0)
signal.signal(signal.SIGINT, cleanup)

# ======== Modelo ========
print("🔍 Cargando modelo de piezas...")
model = tf.keras.models.load_model(MODELO_PATH)

def preprocess_square_bgr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, CASILLA_SIZE, interpolation=cv2.INTER_AREA)
    return np.expand_dims(img_res.astype(np.float32)/255.0, 0)

def predict_label(img_bgr):
    p = model.predict(preprocess_square_bgr(img_bgr), verbose=0)[0]
    return CLASSES[int(np.argmax(p))]

# ======== Utiles detection ========
def _cluster_positions(pos, tol=8):
    if not pos: return []
    pos = sorted(pos)
    groups = []
    cur = [pos[0]]
    for x in pos[1:]:
        if abs(x-cur[-1]) <= tol: cur.append(x)
        else: groups.append(int(round(np.mean(cur)))); cur=[x]
    groups.append(int(round(np.mean(cur))))
    return groups

def _pick_equispaced(positions, n=9):
    positions = sorted(positions)
    if len(positions) < n: return None
    target = np.linspace(positions[0], positions[-1], n)
    chosen = []
    used = set()
    for t in target:
        idx = min(range(len(positions)), key=lambda i: (abs(positions[i]-t), i) if i not in used else (1e9, i))
        used.add(idx); chosen.append(positions[idx])
    chosen = sorted(chosen)
    if len(chosen) != n: return None
    diffs = np.diff(chosen)
    if len(diffs) and np.std(diffs) > UNIFORM_STD_TOL: return None
    return chosen

def _debug_save(name, img):
    cv2.imwrite(name, img)

def try_hough(frame):
    h, w = frame.shape[:2]
    for i,params in enumerate(HOUGH_TRIES):
        low, high = params["canny"]
        thr, gap  = params["thr"], params["gap"]
        min_len   = int(min(h,w) * params["minFrac"])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if params["dilate"]>0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            gray = cv2.dilate(gray, k, iterations=params["dilate"])
        edges = cv2.Canny(gray, low, high)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=thr,
                                minLineLength=min_len, maxLineGap=gap)
        _debug_save(f"_dbg_edges_{i}.png", edges)
        if lines is None: continue

        vert, hori = [], []
        for x1,y1,x2,y2 in lines[:,0,:]:
            dx, dy = x2-x1, y2-y1
            if abs(dx) < abs(dy)*0.35:  vert.append((x1+x2)//2)
            elif abs(dy) < abs(dx)*0.35: hori.append((y1+y2)//2)

        vx = _cluster_positions(vert, tol=10)
        hy = _cluster_positions(hori, tol=10)
        xs = _pick_equispaced(vx, 9)
        ys = _pick_equispaced(hy, 9)
        if xs is None or ys is None: continue

        L,R = xs[0], xs[-1]
        T,B = ys[0], ys[-1]
        W,H = R-L, B-T
        if W<=0 or H<=0: continue
        ratio = W/float(H)
        if not (0.9<=ratio<=1.1): continue
        if W<MIN_BOARD_SIZE or H<MIN_BOARD_SIZE: continue
        return (T,B,L,R,xs,ys)
    return None

def try_max_square_contour(frame):
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(gray,30,120)
    edges = cv2.dilate(edges, np.ones((3,3),np.uint8), 2)
    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best=None;best_area=0
    for c in contours:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4:
            x,y,w,h = cv2.boundingRect(approx)
            area = w*h
            r = w/float(h)
            if area>best_area and 0.85<r<1.15 and w>=MIN_BOARD_SIZE and h>=MIN_BOARD_SIZE:
                best=(y,y+h,x,x+w,[x+i*w//8 for i in range(9)],[y+i*h//8 for i in range(9)])
                best_area=area
    return best

_last_manual_crop = None
def manual_crop_once(frame):
    global _last_manual_crop
    if _last_manual_crop is not None:
        T,B,L,R = _last_manual_crop
        xs = [L + i*(R-L)//8 for i in range(9)]
        ys = [T + i*(B-T)//8 for i in range(9)]
        return (T,B,L,R,xs,ys)

    print("🖱️ Click en esquina SUPERIOR-IZQUIERDA y luego en INFERIOR-DERECHA del tablero. (Ventana: 'Selecciona tablero')")
    tmp = frame.copy()
    pts=[]
    def on_mouse(e,x,y,flags,param):
        if e==cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y))
    cv2.imshow("Selecciona tablero", tmp)
    cv2.setMouseCallback("Selecciona tablero", on_mouse)
    while len(pts)<2:
        if cv2.waitKey(50)&0xFF in (27, ord('q')): break
    cv2.destroyWindow("Selecciona tablero")
    if len(pts)<2: return None
    (L,T),(R,B)=pts[0],pts[1]
    L,R = sorted([L,R]); T,B = sorted([T,B])
    _last_manual_crop = (T,B,L,R)
    xs = [L + i*(R-L)//8 for i in range(9)]
    ys = [T + i*(B-T)//8 for i in range(9)]
    return (T,B,L,R,xs,ys)

def detectar_tablero(frame):
    # 1) Hough multi-try
    d = try_hough(frame)
    if d: return d
    # 2) Contorno máximo casi cuadrado
    d = try_max_square_contour(frame)
    if d: return d
    # 3) Manual una sola vez
    if CLICK_TO_CROP_FALLBACK:
        d = manual_crop_once(frame)
        if d: return d
    return None

# ======== Render ========
def render_labels(grid):
    img = np.zeros((8*SQ,8*SQ,3), np.uint8)
    for r in range(8):
        for c in range(8):
            color = CLR_LIGHT if (r+c)%2==0 else CLR_DARK
            cv2.rectangle(img,(c*SQ,r*SQ),((c+1)*SQ,(r+1)*SQ),color,-1)
    for r in range(8):
        for c in range(8):
            lab = grid[r][c]
            if lab=='empty': continue
            (tw,th),_ = cv2.getTextSize(lab,FONT,0.5,1)
            x = c*SQ + (SQ-tw)//2
            y = r*SQ + (SQ+th)//2
            cv2.putText(img, lab,(x,y),FONT,0.5,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(img, lab,(x,y),FONT,0.5,(255,255,255),1,cv2.LINE_AA)
    return img

def grid_to_fen(grid):
    rows=[]
    for fila in grid:
        s=""; empty=0
        for lab in fila:
            if lab=='empty': empty+=1
            else:
                if empty: s+=str(empty); empty=0
                s+= lab[1].upper() if lab[0]=='w' else lab[1].lower()
        if empty: s+=str(empty)
        rows.append(s)
    turn = "w" if JUEGO_CON_BLANCAS else "b"
    return "/".join(rows) + f" {turn} - - 0 1"

# ======== Pipeline ========
def compute_once():
    # captura
    shot = ImageGrab.grab()
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)

    det = detectar_tablero(frame)
    if not det:
        print("⚠️ No se detectó el tablero automáticamente.")
        return None, None, np.zeros((8*SQ,8*SQ,3),np.uint8)

    T,B,L,R,xs,ys = det
    W,H = R-L, B-T
    print(f"📏 Tablero: L={L} R={R} T={T} B={B}  (W={W}, H={H})")

    # recortes + predicción
    grid = [['' for _ in range(8)] for _ in range(8)]
    for ry in range(8):
        for rx in range(8):
            y1,y2 = ys[ry], ys[ry+1]
            x1,x2 = xs[rx], xs[rx+1]
            cell = frame[y1:y2, x1:x2]
            lab = predict_label(cell)
            if JUEGO_CON_BLANCAS:
                grid[ry][rx] = lab
            else:
                grid[7-ry][7-rx] = lab

    print("\n📷 Posición detectada:")
    for fila in grid: print(" ".join(l if l!='empty' else '.' for l in fila))
    fen = grid_to_fen(grid)
    print(f"\n♟ FEN:\n{fen}")

    vis = render_labels(grid)
    return grid, fen, vis

# ======== MAIN ========
if __name__ == "__main__":
    print(f"⏳ Cambia a la pantalla de Chess.com... Captura en {TIEMPO_ESPERA}s…")
    time.sleep(TIEMPO_ESPERA)

    grid, fen, vis = compute_once()
    cv2.imshow("Detector de tablero — CNN", vis)

    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (27, ord('q')):
            cleanup()
        elif k in (ord('r'), ord('R')):
            print("\n🔄 Refrescando…")
            grid, fen, vis = compute_once()
            cv2.imshow("Detector de tablero — CNN", vis)
