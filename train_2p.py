# train_2p.py
# Entrena el detector 2P (top-left, bottom-right) con la estructura:
# data/images/*.png  y data/labels.jsonl

import os, json, random
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

# ----------------- Config -----------------
IMAGES_DIR   = Path("data/images")
LABELS_PATH  = Path("data/labels.jsonl")   # tl=[x,y], br=[x,y]
INPUT        = 384                         # debe coincidir luego al predecir
BATCH        = 16
EPOCHS       = 10
LR           = 3e-4
VAL_SPLIT    = 0.12                        # 12% validación
USE_PRETRAINED = False                     # True si ya solucionaste certificados
OUT_MODEL    = "board_2p.h5"

# ----------------- Utils ------------------
def letterbox(img, size=INPUT):
    """Resize con padding (sin deformar) a (size,size). Devuelve imagen, escala y offset."""
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(h * s), int(w * s)
    rsz = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = rsz
    return canvas, s, left, top

def load_rows(labels_path=LABELS_PATH):
    rows = []
    with open(labels_path, "r") as f:
        for ln in f:
            rows.append(json.loads(ln))
    # Filtra solo los que existen en disco
    out = []
    for r in rows:
        p = IMAGES_DIR / r["file"]
        if p.exists():
            out.append(r)
    if not out:
        raise SystemExit("❌ No se encontraron imágenes que coincidan con labels.jsonl")
    return out

def normalize_target(w, h, tl, br, size=INPUT):
    """Convierte coords originales a normalizadas [0,1] en el espacio letterbox."""
    lb, s, left, top = letterbox(np.zeros((h, w, 3), np.uint8), size)
    x1 = (tl[0] * s + left) / size
    y1 = (tl[1] * s + top)  / size
    x2 = (br[0] * s + left) / size
    y2 = (br[1] * s + top)  / size
    return np.array([x1, y1, x2, y2], np.float32)

# ----------------- Dataset ----------------
class BoardDataset(tf.keras.utils.Sequence):
    def __init__(self, rows, batch=BATCH, shuffle=True, augment=True):
        self.rows = rows
        self.batch = batch
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return (len(self.rows) + self.batch - 1) // self.batch

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.rows)

    def __getitem__(self, idx):
        batch_rows = self.rows[idx*self.batch:(idx+1)*self.batch]
        xs, ys = [], []
        for r in batch_rows:
            img = cv2.imread(str(IMAGES_DIR / r["file"]))
            if img is None:
                continue
            h, w = img.shape[:2]

            # augment ligero (brightness/contrast, blur, jitter)
            if self.augment:
                if random.random() < 0.5:
                    alpha = 0.85 + 0.4*random.random()
                    beta  = random.randint(-12, 12)
                    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                if random.random() < 0.15:
                    img = cv2.GaussianBlur(img, (3,3), 0)
                # jitter leve de cajas (manteniendo coherencia)
                tl = np.array(r["tl"], np.float32)
                br = np.array(r["br"], np.float32)
                jitter = 0.02 * np.array([w, h], np.float32)
                tl += (np.random.randn(2) * jitter)
                br += (np.random.randn(2) * jitter)
                tl = np.clip(tl, [0,0], [w-2, h-2])
                br = np.clip(br, [tl[0]+2, tl[1]+2], [w-1, h-1])
            else:
                tl = np.array(r["tl"], np.float32)
                br = np.array(r["br"], np.float32)

            lb, _, _, _ = letterbox(img, INPUT)
            xs.append(lb.astype(np.float32)/255.0)
            ys.append(normalize_target(w, h, tl, br, INPUT))

        return np.stack(xs), np.stack(ys)

# ----------------- Modelo -----------------
def build_model():
    if USE_PRETRAINED:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(INPUT, INPUT, 3), include_top=False, weights="imagenet")
        train_base = False
    else:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(INPUT, INPUT, 3), include_top=False, weights=None)
        train_base = True

    inp = tf.keras.Input((INPUT, INPUT, 3))
    x = base(inp, training=train_base)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(4, activation="sigmoid")(x)  # tlx,tly,brx,bry en [0,1]
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss=tf.keras.losses.Huber(),
                  metrics=[])
    return model

# ----------------- Train ------------------
def main():
    rows = load_rows()
    print(f"🗂  Imágenes etiquetadas: {len(rows)}")
    # train/val split
    random.shuffle(rows)
    n_val = max(1, int(len(rows)*VAL_SPLIT))
    val_rows = rows[:n_val]
    tr_rows  = rows[n_val:]

    train_ds = BoardDataset(tr_rows, augment=True, shuffle=True)
    val_ds   = BoardDataset(val_rows, augment=False, shuffle=False)

    model = build_model()
    model.summary()

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        OUT_MODEL, monitor="val_loss", save_best_only=True, verbose=1)
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True)

    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS, callbacks=[ckpt, es])

    # guarda por si acaso (además del best en ckpt)
    model.save(OUT_MODEL)
    print(f"✅ Modelo guardado: {OUT_MODEL}")

if __name__ == "__main__":
    main()
