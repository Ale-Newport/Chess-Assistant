import cv2
import numpy as np
import time
from PIL import ImageGrab
import os
import tablero
import chess.engine
import signal, sys


# ============ LIMPIEZA / CTRL+C ============
def cleanup(*_):
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("\n👋 Cancelado por el usuario. Todo cerrado.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

# ============ CONFIG ============
JUEGO_CON_BLANCAS = True
TIEMPO_ESPERA = 5
TAMANO_PLANTILLA = (150, 150)  # (ancho, alto)
columnas = "abcdefgh"

# Coordenadas del tablero en pantalla (ajústalas)
TOP = 365
BOTTOM = 2100
LEFT = 470
RIGHT = 2200

# Colores de casillas (BGR) del tema clásico de Chess.com
CLR_LIGHT = (211, 236, 235)  # #ebecd3 en BGR
CLR_DARK  = (90, 148, 122)   # #7a945a en BGR

# Ejecutable de Stockfish
STOCKFISH_PATH = "stockfish/src/stockfish"  # o /opt/homebrew/bin/stockfish si usas brew

# ============ UTILES ============
def square_to_rc(square, white_pov=True):
    """Convierte 'e2' -> (row, col) en 0..7 según la orientación."""
    file = columnas.index(square[0])  # a=0..h=7
    rank = int(square[1])             # 1..8
    if white_pov:
        row = 8 - rank
        col = file
    else:
        row = rank - 1
        col = 7 - file
    return row, col

def draw_move(board_img, from_sq, to_sq, sq_w, sq_h):
    """Dibuja flecha y resalta origen/destino."""
    r1, c1 = square_to_rc(from_sq, JUEGO_CON_BLANCAS)
    r2, c2 = square_to_rc(to_sq, JUEGO_CON_BLANCAS)

    # Rectángulos semi-transparentes
    overlay = board_img.copy()
    def fill_sq(r, c, color=(0, 255, 0)):
        x1, y1 = c * sq_w, r * sq_h
        x2, y2 = x1 + sq_w, y1 + sq_h
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    fill_sq(r1, c1, (0, 255, 0))
    fill_sq(r2, c2, (0, 165, 255))
    cv2.addWeighted(overlay, 0.25, board_img, 0.75, 0, board_img)

    # Flecha centro a centro
    p1 = (c1 * sq_w + sq_w // 2, r1 * sq_h + sq_h // 2)
    p2 = (c2 * sq_w + sq_w // 2, r2 * sq_h + sq_h // 2)
    cv2.arrowedLine(board_img, p1, p2, (0, 255, 0), thickness=6, tipLength=0.2)

def limpiar_fondo_y_redimensionar(img_bgr, output_path):
    """Elimina colores de casilla y redimensiona a TAMANO_PLANTILLA. Guarda RGBA."""
    img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    colores_fondo = [np.array(CLR_LIGHT), np.array(CLR_DARK)]
    tolerancia = 30
    for color in colores_fondo:
        dif = np.abs(img_bgr.astype(int) - color)
        mask = np.all(dif < tolerancia, axis=2)
        img_rgba[mask] = [0, 0, 0, 0]
    resized = cv2.resize(img_rgba, TAMANO_PLANTILLA, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized)

def estimar_color_casilla(casilla_rgba):
    """Devuelve 'w', 'b' o None según el brillo de los píxeles no transparentes."""
    if casilla_rgba is None: return None
    if casilla_rgba.shape[2] == 4:
        alpha = casilla_rgba[:, :, 3]
        mask = alpha > 0
        if mask.sum() < 50: return None
        bgr = casilla_rgba[:, :, :3]
        v_mean = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2][mask].mean()
    else:
        v_mean = cv2.cvtColor(casilla_rgba, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
    if v_mean >= 150: return 'w'
    if v_mean <= 115: return 'b'
    return None

def detectar_pieza(casilla_img_rgba, plantillas_w, plantillas_b):
    """Matching en color con máscara alpha. Devuelve 'wP', 'bK'… o '.'"""
    if casilla_img_rgba is None: return '.'
    if casilla_img_rgba.shape[:2] != (TAMANO_PLANTILLA[1], TAMANO_PLANTILLA[0]):
        casilla_img_rgba = cv2.resize(casilla_img_rgba, TAMANO_PLANTILLA, interpolation=cv2.INTER_AREA)
    casilla_bgr = casilla_img_rgba[:, :, :3] if casilla_img_rgba.shape[2]==4 else casilla_img_rgba

    color_est = estimar_color_casilla(casilla_img_rgba)
    candidatos = plantillas_w if color_est=='w' else plantillas_b if color_est=='b' else {**plantillas_w, **plantillas_b}

    mejor, mejor_val = '.', -1.0
    for nombre, data in candidatos.items():
        tpl, m = data["img"], data["mask"]
        res = cv2.matchTemplate(casilla_bgr, tpl, cv2.TM_CCORR_NORMED, mask=m) if m is not None \
              else cv2.matchTemplate(casilla_bgr, tpl, cv2.TM_CCORR_NORMED)
        _, val, _, _ = cv2.minMaxLoc(res)
        if val > mejor_val:
            mejor_val, mejor = val, nombre
    return mejor if mejor_val >= 0.86 else '.'

def grid_a_fen(grid):
    fen_rows = []
    for fila in grid:
        s, empty = "", 0
        for c in fila:
            if c == '.' or c == "":
                vacias = True
                empty += 1
            else:
                if empty: s += str(empty); empty = 0
                s += c[1].upper() if c[0]=='w' else c[1].lower()
        if empty: s += str(empty)
        fen_rows.append(s)
    pos = "/".join(fen_rows)
    turno = "w" if JUEGO_CON_BLANCAS else "b"
    return f"{pos} {turno} - - 0 1"

def cargar_plantillas():
    plantillas_w, plantillas_b = {}, {}
    for archivo in os.listdir("piezas"):
        if not archivo.endswith(".png"):
            continue
        nombre = archivo[:-4]  # wP, bK...
        tpl_rgba = cv2.imread(os.path.join("piezas", archivo), cv2.IMREAD_UNCHANGED)
        if tpl_rgba is None:
            continue
        tpl_rgba = cv2.resize(tpl_rgba, TAMANO_PLANTILLA, interpolation=cv2.INTER_AREA)
        bgr = tpl_rgba[:, :, :3] if tpl_rgba.shape[2] == 4 else tpl_rgba
        mask = (tpl_rgba[:, :, 3] > 0).astype(np.uint8) * 255 if tpl_rgba.shape[2] == 4 else None
        (plantillas_w if nombre[0].lower()=='w' else plantillas_b)[nombre] = {"img": bgr, "mask": mask}
    return plantillas_w, plantillas_b

def render_board(grid, best_move, plantillas_w, plantillas_b):
    sq_w, sq_h = TAMANO_PLANTILLA
    board_img = np.zeros((8*sq_h, 8*sq_w, 3), dtype=np.uint8)

    # Pintar casillas
    for r in range(8):
        for c in range(8):
            color = CLR_LIGHT if (r + c) % 2 == 0 else CLR_DARK
            cv2.rectangle(board_img, (c*sq_w, r*sq_h), ((c+1)*sq_w, (r+1)*sq_h), color, -1)

    # Colocar piezas (máscara alpha)
    plantillas_all = {**plantillas_w, **plantillas_b}
    for r in range(8):
        for c in range(8):
            pieza = grid[r][c]
            if pieza == '.' or pieza == '':
                continue
            tpl = plantillas_all.get(pieza)
            if tpl is None:
                continue
            img_piece = tpl["img"]
            mask      = tpl["mask"]
            y1, y2 = r * sq_h, (r + 1) * sq_h
            x1, x2 = c * sq_w, (c + 1) * sq_w
            roi = board_img[y1:y2, x1:x2]
            cv2.copyTo(img_piece, mask, roi)

    # Dibujar jugada
    if best_move:
        from_sq, to_sq = best_move[:2], best_move[2:4]
        draw_move(board_img, from_sq, to_sq, sq_w, sq_h)

    return board_img

def compute_once(plantillas_w, plantillas_b):
    """Captura pantalla, detecta, calcula FEN y mejor jugada. Devuelve (board_img, best_move)."""
    # Captura y recorte
    screenshot = ImageGrab.grab()
    screenshot.save("pantalla.png")
    img = cv2.imread("pantalla.png")
    if img is None:
        print("❌ Error: No se pudo cargar 'pantalla.png'.")
        return np.zeros((8*TAMANO_PLANTILLA[1], 8*TAMANO_PLANTILLA[0], 3), np.uint8), None

    tablero = img[TOP:BOTTOM, LEFT:RIGHT]
    cv2.imwrite("tablero.png", tablero)

    casilla_h = (BOTTOM - TOP) // 8
    casilla_w = (RIGHT - LEFT) // 8

    # Recortar + limpiar casillas
    os.makedirs("casillas", exist_ok=True)
    for y in range(8):
        for x in range(8):
            casilla = tablero[y*casilla_h:(y+1)*casilla_h, x*casilla_w:(x+1)*casilla_w]
            if JUEGO_CON_BLANCAS:
                col = columnas[x]; row = 8 - y
            else:
                col = columnas[7 - x]; row = y + 1
            ruta = f"casillas/{col}{row}.png"
            img_rgba = cv2.cvtColor(casilla, cv2.COLOR_BGR2BGRA)
            colores_fondo = [np.array(CLR_LIGHT), np.array(CLR_DARK)]
            tolerancia = 30
            for color in colores_fondo:
                dif = np.abs(casilla.astype(int) - color)
                mask = np.all(dif < tolerancia, axis=2)
                img_rgba[mask] = [0, 0, 0, 0]
            resized = cv2.resize(img_rgba, TAMANO_PLANTILLA, interpolation=cv2.INTER_AREA)
            cv2.imwrite(ruta, resized)

    # Construir grid
    grid = [['' for _ in range(8)] for _ in range(8)]
    for y in range(8):
        for x in range(8):
            if JUEGO_CON_BLANCAS:
                grid_y, grid_x = y, x; col = columnas[x]; row = 8 - y
            else:
                grid_y, grid_x = 7 - y, 7 - x; col = columnas[7 - x]; row = y + 1
            cas = cv2.imread(f"casillas/{col}{row}.png", cv2.IMREAD_UNCHANGED)
            grid[grid_y][grid_x] = detectar_pieza(cas, plantillas_w, plantillas_b)

    # Mostrar grid en consola
    print("\n📷 Posición detectada:")
    for fila in grid:
        print(" ".join(p if p != '.' else '.' for p in fila))

    # FEN
    fen = grid_a_fen(grid)
    print(f"\n♟ FEN generado:\n{fen}")

    # Stockfish (sin 'global engine'; usamos context manager)
    best_move = None
    try:
        board = tablero.Board(fen)
        with tablero.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as eng:
            result = eng.play(board, tablero.engine.Limit(time=1.0))
            best_move = result.move.uci()
            print(f"✅ Mejor jugada sugerida: {best_move}")
    except FileNotFoundError:
        print("❌ No se encontró el ejecutable de Stockfish. Revisa STOCKFISH_PATH.")
    except (tablero.engine.EngineError, tablero.engine.EngineTerminatedError) as e:
        print(f"⚠️ Error con el motor Stockfish: {e}")
    except Exception as e:
        print(f"⚠️ Error inesperado con Stockfish: {e}")

    # Render
    board_img = render_board(grid, best_move, plantillas_w, plantillas_b)
    return board_img, best_move

# ============ MAIN ============
if __name__ == "__main__":
    # Espera inicial para que te dé tiempo a cambiar de ventana
    print(f"⏳ Cambia a la pantalla de Chess.com... Captura en {TIEMPO_ESPERA} segundos...")
    time.sleep(TIEMPO_ESPERA)

    # Cargar plantillas una vez
    print("🔍 Cargando plantillas de piezas...")
    plantillas_w, plantillas_b = cargar_plantillas()

    # Primer cálculo
    board_img, best_move = compute_once(plantillas_w, plantillas_b)
    title = f"Asistente Chess — mejor jugada: {best_move}" if best_move else "Asistente Chess"
    cv2.imshow("Asistente Chess", board_img)
    cv2.setWindowTitle("Asistente Chess", title)

    # Bucle de UI: refresca con 'r', salir con q/ESC
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (27, ord('q')):  # ESC o q
            cleanup()
        elif k in (ord('r'), ord('R')):  # refrescar
            print("\n🔄 Refrescando…")
            board_img, best_move = compute_once(plantillas_w, plantillas_b)
            title = f"Asistente Chess — mejor jugada: {best_move}" if best_move else "Asistente Chess"
            cv2.imshow("Asistente Chess", board_img)
            cv2.setWindowTitle("Asistente Chess", title)
