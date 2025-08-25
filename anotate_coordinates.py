import cv2, json
from pathlib import Path

IMG_DIR = Path("data/images")
OUT = Path("data/labels.jsonl")

def main():
    imgs = sorted([p for p in IMG_DIR.glob("*.*") if p.suffix.lower() in [".png",".jpg",".jpeg"]])
    if not imgs:
        print("❌ No hay imágenes en data/images/"); return

    print("🖱 Marca 2 puntos por imagen: 1) Arriba-Izquierda (TL), 2) Abajo-Derecha (BR)")
    with open(OUT, "a") as f:
        for p in imgs:
            img = cv2.imread(str(p))
            disp = img.copy()
            pts = []
            def on_mouse(e,x,y,flags,param):
                nonlocal pts, disp
                if e == cv2.EVENT_LBUTTONDOWN:
                    pts.append((x,y))
                    cv2.circle(disp,(x,y),6,(0,0,255),-1)
                    cv2.imshow(p.name, disp)

            cv2.imshow(p.name, disp)
            cv2.setMouseCallback(p.name, on_mouse)
            while True:
                k = cv2.waitKey(30) & 0xFF
                if k == 27:  # ESC
                    cv2.destroyAllWindows(); return
                if len(pts) == 2:
                    tl, br = pts
                    rec = {"file": p.name, "tl": tl, "br": br}
                    f.write(json.dumps(rec) + "\n")
                    print("✅", rec)
                    break
            cv2.destroyAllWindows()
    print(f"📂 Guardado en {OUT}")

if __name__ == "__main__":
    main()
