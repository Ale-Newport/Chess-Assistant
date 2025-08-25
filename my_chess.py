# chess_gui_click_edit_sf_capture.py
# Ejecutar:
#   python chess_gui_click_edit_sf_capture.py --orient white|black
#
# Teclas:
#   Click->Click para mover · U/Backspace deshacer · Shift+U/Ctrl+Z turno · F voltear · R reiniciar ·
#   E editor · S sugerencia (Shift profundo) · P (en editor) capturar tablero · Q/Esc salir
#
# Requisitos:
#   pip install pygame python-chess pillow tensorflow
#   (en mac Apple Silicon: tensorflow-macos)
#   Stockfish en STOCKFISH_PATH o cambia la ruta
#   Pack de piezas opcional en carpeta ASSETS_DIR (sino usa Unicode)

import sys
import os
import math
import argparse
import numpy as np
import pygame
import chess
import chess.engine
from PIL import ImageGrab, Image

# ---------- Configuración del motor Stockfish ----------
STOCKFISH_PATH = "stockfish/src/stockfish"
ENGINE_TIME_S = 1.0        # tiempo por movimiento cuando pulsas S
ENGINE_TIME_DEEP_S = 3.0   # tiempo cuando pulsas Shift+S

# ---------- Modelo de reconocimiento de piezas ----------
MODEL_PATH = "modelo_piezas.h5"
MODEL_INPUT_SIZE = 64
LABELS = ["empty","wp","wn","wb","wr","wq","wk","bp","bn","bb","br","bq","bk"]
LABEL_TO_SYM = {
    "wp":"P","wn":"N","wb":"B","wr":"R","wq":"Q","wk":"K",
    "bp":"p","bn":"n","bb":"b","br":"r","bq":"q","bk":"k"
}
piece_model = None  # se carga bajo demanda

# ----------------- Tema (estilo chess.com) -----------------
LIGHT_SQ = pygame.Color("#EEEED2")
DARK_SQ  = pygame.Color("#769656")
HIGHLIGHT_LAST = pygame.Color("#F6F669")
HIGHLIGHT_SEL  = pygame.Color("#BACA44")
MOVE_DOT       = pygame.Color(0, 0, 0, 120)
CAPTURE_RING   = pygame.Color(0, 0, 0, 200)
LABEL_COLOR_LIGHT = pygame.Color(60, 60, 60)
BG_COLOR = pygame.Color("#1b1f22")

BOARD_TILES = 8
SQUARE_SIZE = 64     # tamaño lógico: con SCALED se adapta a la ventana
MARGIN = 40
BOARD_PIX = BOARD_TILES * SQUARE_SIZE

# Panel derecho para edición
SIDE_W = 220
INNER_PAD = 16

WIN_W = BOARD_PIX + 2 * MARGIN + SIDE_W
WIN_H = BOARD_PIX + 2 * MARGIN
FPS = 60

ASSETS_DIR = "piezas"  # pon aquí tus PNG (wp.png, bp.png, ...); si faltan usa Unicode

PIECE_FILES = {
    'P': 'wp.png', 'p': 'bp.png',
    'N': 'wn.png', 'n': 'bn.png',
    'B': 'wb.png', 'b': 'bb.png',
    'R': 'wr.png', 'r': 'br.png',
    'Q': 'wq.png', 'q': 'bq.png',
    'K': 'wk.png', 'k': 'bk.png',
}

UNICODE_GLYPHS = {
    'P': '♙', 'p': '♟',
    'N': '♘', 'n': '♞',
    'B': '♗', 'b': '♝',
    'R': '♖', 'r': '♜',
    'Q': '♕', 'q': '♛',
    'K': '♔', 'k': '♚',
}

# ---------- Estado del motor / sugerencias ----------
engine = None
engine_err = ""
hint_move = None       # chess.Move sugerido
hint_text = ""         # texto como "SF: e2e4 (+0.35) para Blancas"
hint_deadline_ms = 0   # hasta cuándo mostrar la sugerencia


# ========================= Utilidades generales =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orient", choices=["white", "black"], default="white",
                    help="Perspectiva inicial: 'white' (por defecto) o 'black'.")
    return ap.parse_args()

def square_at(col, row, white_bottom=True):
    """(col,row) de pantalla -> índice square (0..63)."""
    if white_bottom:
        file_ = col
        rank_ = 7 - row
    else:
        file_ = 7 - col
        rank_ = row
    return chess.square(file_, rank_)

def col_row_from_square(square, white_bottom=True):
    """square (0..63) -> (col,row) de pantalla."""
    file_ = chess.square_file(square)
    rank_ = chess.square_rank(square)
    if white_bottom:
        col = file_
        row = 7 - rank_
    else:
        col = 7 - file_
        row = rank_
    return col, row

def load_piece_surfaces(square_px):
    size = int(square_px * 0.84)
    cache = {}
    ok_png = os.path.isdir(ASSETS_DIR) and any(
        os.path.isfile(os.path.join(ASSETS_DIR, f)) for f in PIECE_FILES.values()
    )
    if ok_png:
        for sym, fname in PIECE_FILES.items():
            path = os.path.join(ASSETS_DIR, fname)
            if os.path.isfile(path):
                img = pygame.image.load(path).convert_alpha()
                cache[sym] = pygame.transform.smoothscale(img, (size, size))
    missing = [s for s in UNICODE_GLYPHS.keys() if s not in cache]
    if missing:
        pygame.font.init()
        try:
            font = pygame.font.SysFont("DejaVu Sans", size)
        except Exception:
            font = pygame.font.Font(None, size)
        for s in missing:
            glyph = UNICODE_GLYPHS[s]
            color = (240, 240, 240) if s.isupper() else (20, 20, 20)
            surf = font.render(glyph, True, color)
            canvas = pygame.Surface((size, size), pygame.SRCALPHA)
            rect = surf.get_rect(center=(size // 2, size // 2))
            canvas.blit(surf, rect)
            cache[s] = canvas
    return cache

def draw_labels(screen, white_bottom=True):
    files = "abcdefgh" if white_bottom else "hgfedcba"
    ranks = "12345678" if white_bottom else "87654321"
    pygame.font.init()
    font = pygame.font.SysFont("Inter,Arial,Helvetica", 16)
    for i, f in enumerate(files):
        x = MARGIN + i * SQUARE_SIZE + 4
        y = MARGIN + BOARD_PIX + 4
        screen.blit(font.render(f, True, LABEL_COLOR_LIGHT), (x, y))
    for i, r in enumerate(reversed(ranks)):
        x = 16
        y = MARGIN + i * SQUARE_SIZE + 4
        screen.blit(font.render(r, True, LABEL_COLOR_LIGHT), (x, y))

def draw_board(screen, board, imgs, white_bottom, selected_sq=None,
               legal_to=None, last_move=None):
    screen.fill(BG_COLOR)
    # Casillas + resaltes
    for row in range(BOARD_TILES):
        for col in range(BOARD_TILES):
            x = MARGIN + col * SQUARE_SIZE
            y = MARGIN + row * SQUARE_SIZE
            sq = square_at(col, row, white_bottom)
            is_light = (col + row) % 2 == 0
            color = LIGHT_SQ if is_light else DARK_SQ

            if last_move and (sq == last_move.from_square or sq == last_move.to_square):
                pygame.draw.rect(screen, HIGHLIGHT_LAST, (x, y, SQUARE_SIZE, SQUARE_SIZE))
            else:
                pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

            if selected_sq == sq:
                pygame.draw.rect(screen, HIGHLIGHT_SEL, (x, y, SQUARE_SIZE, SQUARE_SIZE), 4)

    # Dots / anillos de jugadas legales
    if selected_sq is not None and legal_to:
        for to_sq, is_capture in legal_to:
            col, row = col_row_from_square(to_sq, white_bottom)
            cx = MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2
            cy = MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2
            if is_capture:
                outer = SQUARE_SIZE // 2 - 8
                inner = outer - 6
                pygame.draw.circle(screen, CAPTURE_RING, (cx, cy), outer, 4)
                pygame.draw.circle(screen, CAPTURE_RING, (cx, cy), inner, 2)
            else:
                radius = SQUARE_SIZE // 8
                dot = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(dot, MOVE_DOT, (radius, radius), radius)
                screen.blit(dot, (cx - radius, cy - radius))

    # Piezas
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            sym = piece.symbol()
            img = imgs[sym]
            col, row = col_row_from_square(sq, white_bottom)
            x = MARGIN + col * SQUARE_SIZE + (SQUARE_SIZE - img.get_width()) // 2
            y = MARGIN + row * SQUARE_SIZE + (SQUARE_SIZE - img.get_height()) // 2
            screen.blit(img, (x, y))

    draw_labels(screen, white_bottom)

def draw_move_arrow(screen, move: chess.Move, white_bottom: bool, color=(31, 111, 235), thickness=10):
    # centros de casillas
    def center_of(square):
        col, row = col_row_from_square(square, white_bottom)
        cx = MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2
        cy = MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2
        return cx, cy

    start = center_of(move.from_square)
    end   = center_of(move.to_square)

    pygame.draw.line(screen, color, start, end, thickness)

    ang = math.atan2(end[1]-start[1], end[0]-start[0])
    head_len = max(14, SQUARE_SIZE // 4)
    head_w   = max(10, SQUARE_SIZE // 5)
    left = (end[0] - head_len*math.cos(ang) + head_w*math.sin(ang),
            end[1] - head_len*math.sin(ang) - head_w*math.cos(ang))
    right= (end[0] - head_len*math.cos(ang) - head_w*math.sin(ang),
            end[1] - head_len*math.sin(ang) + head_w*math.cos(ang))
    pygame.draw.polygon(screen, color, [end, left, right])

# ========================= Motor (Stockfish) =========================
def start_engine():
    global engine, engine_err
    if engine is not None:
        return True
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"Threads": 2})  # ajusta si quieres
        engine_err = ""
        return True
    except Exception as e:
        engine = None
        engine_err = f"No se pudo iniciar Stockfish: {e}"
        return False

def stop_engine():
    global engine
    try:
        if engine is not None:
            engine.quit()
    finally:
        engine = None

def _score_to_str(score_obj, pov_white: bool) -> str:
    s = score_obj.pov(chess.WHITE if pov_white else chess.BLACK)
    if s.is_mate():
        m = s.mate()
        return f"#{m}" if m is not None else "#?"
    cp = s.score(mate_score=100000)
    if cp is None:
        return "N/A"
    return f"{cp/100:+.2f}"

def suggest_best_move(board: chess.Board, time_s: float):
    """Devuelve (ok, move, text)"""
    if not start_engine():
        return False, None, engine_err or "Stockfish no disponible"
    try:
        info = engine.analyse(board, chess.engine.Limit(time=time_s))
        pv = info.get("pv", [])
        best = pv[0] if pv else None
        if best is None:
            best = engine.play(board, chess.engine.Limit(time=time_s)).move
        score = info.get("score")
        side = "Blancas" if board.turn else "Negras"
        if score is not None:
            score_str = _score_to_str(score, board.turn)
            txt = f"SF: {best.uci()} ({score_str}) para {side}"
        else:
            txt = f"SF: {best.uci()} para {side}"
        return True, best, txt
    except Exception as e:
        return False, None, f"Error al analizar: {e}"

# ========================= Modo edición =========================
class EditState:
    def __init__(self):
        self.active = False
        self.curr_sym = None  # None = borrador, o 'P','n', etc.
        self.white_to_move = True
        self.castling = set()  # {'K','Q','k','q'}
        self.message = ""      # avisos (errores FEN, etc.)

    def from_board(self, board: chess.Board):
        self.white_to_move = board.turn
        self.castling = set()
        if board.has_kingside_castling_rights(chess.WHITE): self.castling.add('K')
        if board.has_queenside_castling_rights(chess.WHITE): self.castling.add('Q')
        if board.has_kingside_castling_rights(chess.BLACK): self.castling.add('k')
        if board.has_queenside_castling_rights(chess.BLACK): self.castling.add('q')
        self.curr_sym = None
        self.message = ""

    def castling_str(self):
        s = "".join(c for c in "KQkq" if c in self.castling)
        return s if s else "-"

def draw_edit_panel(screen, imgs, edit: EditState):
    # Fondo panel
    panel_x = MARGIN + BOARD_PIX + 8
    panel_y = MARGIN
    panel_w = SIDE_W - 16
    panel_h = BOARD_PIX
    pygame.draw.rect(screen, pygame.Color("#21262b"), (panel_x, panel_y, panel_w, panel_h), border_radius=12)
    pygame.draw.rect(screen, pygame.Color("#8fa67a"), (panel_x, panel_y, panel_w, panel_h), 2, border_radius=12)

    font = pygame.font.SysFont("Inter,Arial,Helvetica", 18, bold=True)
    small = pygame.font.SysFont("Inter,Arial,Helvetica", 16)

    # Título
    title = font.render("Modo edición", True, pygame.Color("#EEEED2"))
    screen.blit(title, (panel_x + INNER_PAD, panel_y + INNER_PAD))

    # Fila por tipo: (Nombre, Símbolo blanco, Símbolo negro). None = borrador
    rows = [
        ("Borrar", None, None),
        ("Peón",   'P', 'p'),
        ("Caballo",'N', 'n'),
        ("Alfil",  'B', 'b'),
        ("Torre",  'R', 'r'),
        ("Dama",   'Q', 'q'),
        ("Rey",    'K', 'k'),
    ]

    # Geometría paleta
    box = 44
    gap = 8
    start_y = panel_y + 44
    label_x = panel_x + INNER_PAD
    icon_x1 = panel_x + panel_w - INNER_PAD - box*2 - gap
    icon_x2 = icon_x1 + box + gap

    palette_rects = []
    for i, (name, sym_w, sym_b) in enumerate(rows):
        y = start_y + i * (box + gap)

        # Etiqueta
        screen.blit(small.render(name, True, pygame.Color("#EEEED2")), (label_x, y + 10))

        # Caja izquierda (blancas o borrador)
        r1 = pygame.Rect(icon_x1, y, box, box)
        pygame.draw.rect(screen, pygame.Color("#2b2f33"), r1, border_radius=8)
        pygame.draw.rect(screen, pygame.Color("#8fa67a"), r1, 1, border_radius=8)
        if sym_w is None:
            # Borrador: dibujar una cruz
            pygame.draw.line(screen, pygame.Color("#EEEED2"), (r1.x+8, r1.y+8), (r1.right-8, r1.bottom-8), 2)
            pygame.draw.line(screen, pygame.Color("#EEEED2"), (r1.right-8, r1.y+8), (r1.x+8, r1.bottom-8), 2)
        else:
            img = imgs[sym_w]
            s = pygame.transform.smoothscale(img, (32, 32))
            screen.blit(s, (r1.x + (box-32)//2, r1.y + (box-32)//2))

        # Caja derecha (negras; vacío si es borrador)
        r2 = pygame.Rect(icon_x2, y, box, box)
        pygame.draw.rect(screen, pygame.Color("#2b2f33"), r2, border_radius=8)
        pygame.draw.rect(screen, pygame.Color("#8fa67a"), r2, 1, border_radius=8)
        if sym_b is not None:
            img2 = imgs[sym_b]
            s2 = pygame.transform.smoothscale(img2, (32, 32))
            screen.blit(s2, (r2.x + (box-32)//2, r2.y + (box-32)//2))

        # Resalte de selección
        if edit.curr_sym is None and sym_w is None:
            pygame.draw.rect(screen, pygame.Color("#F6F669"), r1, 3, border_radius=8)
        if sym_w is not None and edit.curr_sym == sym_w:
            pygame.draw.rect(screen, pygame.Color("#F6F669"), r1, 3, border_radius=8)
        if sym_b is not None and edit.curr_sym == sym_b:
            pygame.draw.rect(screen, pygame.Color("#F6F669"), r2, 3, border_radius=8)

        # Hitboxes paleta
        palette_rects.append((r1, sym_w))
        palette_rects.append((r2, sym_b))

    # Turno
    after_palette_y = start_y + len(rows) * (box + gap)
    turn_rect = pygame.Rect(panel_x + INNER_PAD, after_palette_y + 8, panel_w - 2*INNER_PAD, 36)
    pygame.draw.rect(screen, pygame.Color("#2b2f33"), turn_rect, border_radius=10)
    pygame.draw.rect(screen, pygame.Color("#8fa67a"), turn_rect, 2, border_radius=10)
    tlabel = f"Turno: {'Blancas' if edit.white_to_move else 'Negras'} (click)"
    screen.blit(small.render(tlabel, True, pygame.Color("#EEEED2")), (turn_rect.x + 10, turn_rect.y + 8))

    # Enroques
    cast_y = turn_rect.bottom + 10
    screen.blit(small.render("Enroques: (click para alternar)", True, pygame.Color("#EEEED2")),
                (panel_x + INNER_PAD, cast_y))
    cast_y += 6
    btn_sz = 34
    castings = [('K','K'), ('Q','Q'), ('k','k'), ('q','q')]
    cast_rects = []
    for i, (txt, flag) in enumerate(castings):
        r = pygame.Rect(panel_x + INNER_PAD + i*(btn_sz+8), cast_y + 16, btn_sz, btn_sz)
        pygame.draw.rect(screen, pygame.Color("#2b2f33"), r, border_radius=8)
        on = flag in edit.castling
        pygame.draw.rect(screen, pygame.Color("#8fa67a") if on else pygame.Color("#647c59"),
                         r, 2, border_radius=8)
        screen.blit(font.render(txt, True, pygame.Color("#EEEED2")), (r.x+8, r.y+4))
        cast_rects.append((r, flag))

    # Botones acción
    base_y = cast_y + 16 + btn_sz + 10
    def draw_btn(text, y):
        rect = pygame.Rect(panel_x + INNER_PAD, y, panel_w - 2*INNER_PAD, 36)
        pygame.draw.rect(screen, pygame.Color("#2b2f33"), rect, border_radius=10)
        pygame.draw.rect(screen, pygame.Color("#8fa67a"), rect, 2, border_radius=10)
        screen.blit(small.render(text, True, pygame.Color("#EEEED2")), (rect.x + 10, rect.y + 8))
        return rect
    btn_clear  = draw_btn("Limpiar", base_y)
    btn_start  = draw_btn("Inicial", base_y + 44)
    btn_apply  = draw_btn("Terminar edición (E)", base_y + 88)

    # Mensaje de error/info
    if edit.message:
        msg_rect = pygame.Rect(panel_x + INNER_PAD, btn_apply.bottom + 10, panel_w - 2*INNER_PAD, 60)
        pygame.draw.rect(screen, pygame.Color("#3a2a2a"), msg_rect, border_radius=10)
        pygame.draw.rect(screen, pygame.Color("#a36f6f"), msg_rect, 2, border_radius=10)
        wrapped = edit.message[:120]
        screen.blit(small.render(wrapped, True, pygame.Color("#F0D0D0")), (msg_rect.x + 10, msg_rect.y + 8))

    return {
        "turn_rect": turn_rect,
        "cast_rects": cast_rects,
        "btn_clear": btn_clear,
        "btn_start": btn_start,
        "btn_apply": btn_apply,
        "palette_rects": palette_rects,
        "panel_bounds": pygame.Rect(panel_x, panel_y, panel_w, panel_h),
    }

def apply_edit(board: chess.Board, edit: EditState):
    # Construir FEN desde el contenido del tablero actual y opciones elegidas
    pieces = board.board_fen()
    side = 'w' if edit.white_to_move else 'b'
    cast = "".join(c for c in "KQkq" if c in edit.castling) or "-"
    fen = f"{pieces} {side} {cast} - 0 1"
    try:
        board.set_fen(fen)
        return True, ""
    except Exception as e:
        return False, f"Posición inválida: {e}"

# ========================= Captura y análisis =========================
class CaptureState:
    def __init__(self):
        self.active = False
        self.screenshot_pil = None
        self.surf_scaled = None
        self.scale = 1.0
        self.view_rect = pygame.Rect(0,0,0,0)  # dónde se dibuja la screenshot en la ventana
        self.crop_rect = None
        self.dragging = False
        self.drag_start = (0,0)

def _pil_to_surface_scaled(pil_img, max_w, max_h):
    """Devuelve (surf_escalada, view_rect, scale) para encajar la captura en la ventana."""
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h)
    draw_w, draw_h = max(1, int(w * scale)), max(1, int(h * scale))
    surf = pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode).convert()
    surf = pygame.transform.smoothscale(surf, (draw_w, draw_h))
    x = (max_w - draw_w) // 2
    y = (max_h - draw_h) // 2
    return surf, pygame.Rect(x, y, draw_w, draw_h), scale

def begin_capture(capture: 'CaptureState', edit: 'EditState'):
    try:
        img = ImageGrab.grab()  # macOS: requiere permiso de grabación de pantalla
    except Exception as e:
        edit.message = f"No se pudo capturar pantalla: {e}. Concede permiso de 'Screen Recording' a Python."
        return False
    capture.active = True
    capture.screenshot_pil = img.convert("RGB")
    capture.surf_scaled, vr, sc = _pil_to_surface_scaled(capture.screenshot_pil, WIN_W, WIN_H)
    capture.view_rect, capture.scale = vr, sc
    capture.crop_rect = None
    capture.dragging = False
    return True

def draw_capture_ui(screen, capture: 'CaptureState'):
    screen.fill((12,12,12))
    # screenshot escalada
    screen.blit(capture.surf_scaled, capture.view_rect.topleft)
    # overlay oscuro con "hueco" en la selección
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    overlay.fill((0,0,0,120))
    if capture.crop_rect:
        pygame.draw.rect(overlay, (0,0,0,0), capture.crop_rect, 0)
        pygame.draw.rect(screen, (255,215,0), capture.crop_rect, 3, border_radius=6)
    screen.blit(overlay, (0,0))
    # instrucciones
    f = pygame.font.SysFont("Inter,Arial,Helvetica", 18)
    msg = "CAPTURA: arrastra para seleccionar el tablero • Enter: analizar • Esc/P: cancelar"
    screen.blit(f.render(msg, True, (230,230,230)), (MARGIN, 8))

def ensure_model_loaded(edit: 'EditState'):
    global piece_model
    if piece_model is not None:
        return True
    try:
        from tensorflow.keras.models import load_model
        piece_model = load_model(MODEL_PATH)
        return True
    except Exception as e:
        edit.message = f"No pude cargar {MODEL_PATH}: {e}"
        return False

def analyze_cropped_board(board: chess.Board, capture: 'CaptureState', white_bottom: bool, edit: 'EditState'):
    if not capture.crop_rect:
        edit.message = "No hay recorte seleccionado."
        return False
    if not ensure_model_loaded(edit):
        return False

    # Convertir rectángulo en la ventana -> coordenadas de la captura original
    rx = (capture.crop_rect.x - capture.view_rect.x) / capture.scale
    ry = (capture.crop_rect.y - capture.view_rect.y) / capture.scale
    rw = capture.crop_rect.w / capture.scale
    rh = capture.crop_rect.h / capture.scale

    rx, ry = max(0, int(rx)), max(0, int(ry))
    rw, rh = max(1, int(rw)), max(1, int(rh))
    region = capture.screenshot_pil.crop((rx, ry, rx+rw, ry+rh))

    # Dividir en 8x8
    W, H = region.size
    tile_w, tile_h = W / 8.0, H / 8.0

    batch = np.zeros((64, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3), dtype=np.float32)
    idx = 0
    for r in range(8):
        for c in range(8):
            x0 = int(round(c * tile_w))
            y0 = int(round(r * tile_h))
            x1 = int(round((c+1) * tile_w))
            y1 = int(round((r+1) * tile_h))
            tile = region.crop((x0, y0, x1, y1)).resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)
            arr = np.asarray(tile, dtype=np.float32) / 255.0
            batch[idx] = arr
            idx += 1

    # Predicción
    try:
        preds = piece_model.predict(batch, verbose=0)
    except Exception as e:
        edit.message = f"Error al inferir: {e}"
        return False

    # Mapear predicciones a tablero
    classes = preds.argmax(axis=1)
    board.clear()
    idx = 0
    for r in range(8):
        for c in range(8):
            label = LABELS[classes[idx]]
            idx += 1
            if label == "empty":
                continue
            sym = LABEL_TO_SYM.get(label)
            if not sym:
                continue
            ptype = {
                'p': chess.PAWN, 'n': chess.KNIGHT, 'b': chess.BISHOP,
                'r': chess.ROOK, 'q': chess.QUEEN, 'k': chess.KING
            }[sym.lower()]
            color = sym.isupper()
            # r=0 es fila superior del recorte; usa orientación actual
            sq = square_at(c, r, white_bottom)
            board.set_piece_at(sq, chess.Piece(ptype, color))

    edit.message = "Tablero importado desde captura ✓ (revisa y ajusta si hace falta)."
    return True

# ========================= Movimiento y promoción =========================
def legal_moves_from(board, from_sq):
    out = []
    for m in board.legal_moves:
        if m.from_square == from_sq:
            out.append((m.to_square, board.is_capture(m)))
    return out

def needs_promotion(board, move):
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN:
        return False
    to_rank = chess.square_rank(move.to_square)
    return (piece.color and to_rank == 7) or ((not piece.color) and to_rank == 0)

def promotion_selector(screen, white_bottom, to_sq, side_color):
    opts = [("Dama", chess.QUEEN), ("Torre", chess.ROOK),
            ("Alfil", chess.BISHOP), ("Caballo", chess.KNIGHT)]
    col, row = col_row_from_square(to_sq, white_bottom)
    px = MARGIN + col * SQUARE_SIZE + SQUARE_SIZE + 8
    py = MARGIN + row * SQUARE_SIZE - (len(opts) * 44) // 2
    bw, bh = 120, 36
    font = pygame.font.SysFont("Inter,Arial,Helvetica", 20, bold=True)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return None
                mapping = {pygame.K_q: chess.QUEEN, pygame.K_r: chess.ROOK,
                           pygame.K_b: chess.BISHOP, pygame.K_n: chess.KNIGHT}
                if event.key in mapping:
                    return mapping[event.key]
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                for i, (name, val) in enumerate(opts):
                    rect = pygame.Rect(px, py + i * (bh + 8), bw, bh)
                    if rect.collidepoint(mx, my):
                        return val
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 80))
        screen.blit(overlay, (0, 0))
        for i, (name, _) in enumerate(opts):
            rect = pygame.Rect(px, py + i * (bh + 8), bw, bh)
            pygame.draw.rect(screen, pygame.Color("#2b2f33"), rect, border_radius=10)
            pygame.draw.rect(screen, pygame.Color("#8fa67a"), rect, 2, border_radius=10)
            label = font.render(name, True, pygame.Color("#EEEED2"))
            screen.blit(label, (rect.x + 10, rect.y + 6))
        tip = pygame.font.SysFont("Inter,Arial,Helvetica", 16).render(
            "Promociona: Q/R/B/N o click", True, pygame.Color(230, 230, 230))
        screen.blit(tip, (px - 6, py - 28))
        pygame.display.flip()
        pygame.time.Clock().tick(60)

# ========================= Main loop =========================
def main():
    global hint_move, hint_text, hint_deadline_ms
    args = parse_args()
    white_bottom = (args.orient == "white")

    pygame.init()
    flags = pygame.SCALED | pygame.RESIZABLE | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((WIN_W, WIN_H), flags)
    pygame.display.set_caption("Chess.com A.I.")
    clock = pygame.time.Clock()
    font_help = pygame.font.SysFont("Inter,Arial,Helvetica", 16)

    imgs = load_piece_surfaces(SQUARE_SIZE)
    board = chess.Board()

    selected_sq = None
    legal_to = None
    last_move = None

    edit = EditState()
    panel_hit = None  # hitboxes del panel en modo edición
    capture = CaptureState()

    running = True
    while running:
        for event in pygame.event.get():
            # --- CAPTURA ACTIVA: prioridad ---
            if capture.active:
                if event.type == pygame.QUIT:
                    capture.active = False
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_p):
                        capture.active = False  # cancelar
                    elif event.key == pygame.K_RETURN:
                        ok = analyze_cropped_board(board, capture, white_bottom, edit)
                        capture.active = False
                        # seguimos en edición para poder corregir
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if capture.view_rect.collidepoint(mx, my):
                        capture.dragging = True
                        capture.drag_start = (mx, my)
                        capture.crop_rect = pygame.Rect(mx, my, 0, 0)
                elif event.type == pygame.MOUSEMOTION and capture.dragging:
                    mx, my = event.pos
                    x0, y0 = capture.drag_start
                    x1 = max(capture.view_rect.left, min(mx, capture.view_rect.right))
                    y1 = max(capture.view_rect.top,  min(my, capture.view_rect.bottom))
                    x = min(x0, x1); y = min(y0, y1)
                    w = abs(x1 - x0); h = abs(y1 - y0)
                    capture.crop_rect = pygame.Rect(x, y, w, h)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and capture.dragging:
                    capture.dragging = False
                continue  # no procesar más eventos si está en captura

            # --- Resto de eventos ---
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_f:
                    white_bottom = not white_bottom
                elif event.key == pygame.K_r and not edit.active:
                    board.reset()
                    selected_sq = None
                    legal_to = None
                    last_move = None
                elif event.key in (pygame.K_u, pygame.K_BACKSPACE) and not edit.active:
                    if board.move_stack:
                        board.pop()
                        last_move = board.move_stack[-1] if board.move_stack else None
                elif event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL) and not edit.active:
                    if len(board.move_stack) >= 2:
                        board.pop(); board.pop()
                        last_move = board.move_stack[-1] if board.move_stack else None
                    elif board.move_stack:
                        board.pop()
                        last_move = board.move_stack[-1] if board.move_stack else None
                elif event.key == pygame.K_u and (pygame.key.get_mods() & pygame.KMOD_SHIFT) and not edit.active:
                    if len(board.move_stack) >= 2:
                        board.pop(); board.pop()
                        last_move = board.move_stack[-1] if board.move_stack else None
                elif event.key == pygame.K_s and not edit.active:
                    # Sugerir mejor jugada (rápido o profundo con Shift)
                    ok, mv, txt = suggest_best_move(
                        board,
                        ENGINE_TIME_S if not (pygame.key.get_mods() & pygame.KMOD_SHIFT) else ENGINE_TIME_DEEP_S
                    )
                    if ok:
                        hint_move = mv
                        hint_text = txt
                        hint_deadline_ms = pygame.time.get_ticks() + 6000
                        selected_sq = mv.from_square
                        legal_to = [(mv.to_square, board.is_capture(mv))]
                    else:
                        hint_move = None
                        hint_text = txt
                        hint_deadline_ms = pygame.time.get_ticks() + 6000

                elif event.key == pygame.K_e:
                    # toggle edición: si entramos, cargar estado; si salimos, aplicar
                    if not edit.active:
                        edit.active = True
                        edit.from_board(board)
                        selected_sq = None
                        legal_to = None
                    else:
                        ok, msg = apply_edit(board, edit)
                        if ok:
                            edit.active = False
                            edit.message = ""
                            selected_sq = None
                            legal_to = None
                            last_move = None
                        else:
                            edit.message = msg  # mantener activo hasta corregir

                elif event.key == pygame.K_p and edit.active and not capture.active:
                    begin_capture(capture, edit)

                elif event.type == pygame.VIDEORESIZE:
                    # Con SCALED no hace falta recalcular nada
                    pass

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # ¿click en tablero?
                board_area = pygame.Rect(MARGIN, MARGIN, BOARD_PIX, BOARD_PIX)
                if board_area.collidepoint(mx, my):
                    col = (mx - MARGIN) // SQUARE_SIZE
                    row = (my - MARGIN) // SQUARE_SIZE
                    if 0 <= col < 8 and 0 <= row < 8:
                        sq = square_at(col, row, white_bottom)

                        if edit.active:
                            # colocar/borrar pieza
                            if edit.curr_sym is None:
                                board.remove_piece_at(sq)
                            else:
                                color = edit.curr_sym.isupper()
                                ptype = {
                                    'p': chess.PAWN, 'n': chess.KNIGHT, 'b': chess.BISHOP,
                                    'r': chess.ROOK, 'q': chess.QUEEN, 'k': chess.KING
                                }[edit.curr_sym.lower()]
                                board.set_piece_at(sq, chess.Piece(ptype, color))
                        else:
                            # juego normal: click-to-move
                            piece = board.piece_at(sq)
                            if selected_sq is None:
                                if piece and piece.color == board.turn:
                                    selected_sq = sq
                                    legal_to = legal_moves_from(board, selected_sq)
                            else:
                                # si clicas otra pieza del mismo color, cambias selección
                                if piece and piece.color == board.turn and sq != selected_sq:
                                    selected_sq = sq
                                    legal_to = legal_moves_from(board, selected_sq)
                                else:
                                    move = chess.Move(selected_sq, sq)
                                    if needs_promotion(board, move):
                                        promo = promotion_selector(screen, white_bottom, sq, board.turn)
                                        if promo is not None:
                                            move = chess.Move(selected_sq, sq, promotion=promo)
                                        else:
                                            move = None
                                    if move and move in board.legal_moves:
                                        board.push(move)
                                        last_move = move
                                    selected_sq = None
                                    legal_to = None

                # ¿click en panel de edición?
                if edit.active and panel_hit:
                    # paleta
                    for r, sym in panel_hit["palette_rects"]:
                        if r.collidepoint(mx, my):
                            edit.curr_sym = sym
                            break
                    # turno
                    if panel_hit["turn_rect"].collidepoint(mx, my):
                        edit.white_to_move = not edit.white_to_move
                    # enroques
                    for r, flag in panel_hit["cast_rects"]:
                        if r.collidepoint(mx, my):
                            if flag in edit.castling: edit.castling.remove(flag)
                            else: edit.castling.add(flag)
                    # botones
                    if panel_hit["btn_clear"].collidepoint(mx, my):
                        board.clear()
                    if panel_hit["btn_start"].collidepoint(mx, my):
                        board.reset()
                        edit.from_board(board)
                    if panel_hit["btn_apply"].collidepoint(mx, my):
                        ok, msg = apply_edit(board, edit)
                        if ok:
                            edit.active = False
                            edit.message = ""
                            last_move = None
                        else:
                            edit.message = msg

        # --- Dibujo ---
        if capture.active:
            draw_capture_ui(screen, capture)
        else:
            draw_board(screen, board, imgs, white_bottom, selected_sq, legal_to, last_move)

            # Sugerencia del motor (si hay)
            if hint_move is not None:
                draw_move_arrow(screen, hint_move, white_bottom)
                bar = pygame.font.SysFont("Inter,Arial,Helvetica", 16).render(hint_text, True, pygame.Color(230,230,255))
                pad = 6
                rect = bar.get_rect()
                rect.x = MARGIN
                rect.y = MARGIN - rect.height - 8
                bg = pygame.Surface((rect.width + pad*2, rect.height + pad*2), pygame.SRCALPHA)
                bg.fill((0,0,0,160))
                screen.blit(bg, (rect.x - pad, rect.y - pad))
                screen.blit(bar, (rect.x, rect.y))

            if edit.active:
                panel_hit = draw_edit_panel(screen, imgs, edit)
            else:
                panel_hit = None

            # Barra de ayuda
            help_text = (
                "Click para mover · U/Backspace deshacer · Shift+U/Ctrl+Z turno · F voltear · R reiniciar · "
                "S sugerir (Shift profundo) · E editar · Q salir"
                if not edit.active else
                "Edición: click coloca piezas • P capturar tablero • Turno/Enroques en panel • Limpiar/Inicial/Terminar • E aplicar"
            )
            tip = font_help.render(help_text, True, pygame.Color(220, 220, 220))
            pygame.draw.rect(screen, pygame.Color(0, 0, 0, 0), (0, WIN_H - 24, WIN_W, 24))
            screen.blit(tip, (MARGIN, WIN_H - 22))

            # Ocultar sugerencia tras 6s
            if hint_move is not None and pygame.time.get_ticks() > hint_deadline_ms:
                hint_move = None
                hint_text = ""

        pygame.display.flip()
        clock.tick(FPS)

    stop_engine()
    pygame.quit()

if __name__ == "__main__":
    main()
