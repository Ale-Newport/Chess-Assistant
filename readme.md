Chess Assistant 🏁♟️
A Python-based Chess GUI with three main features:
Play Mode: Click-to-move interface with Stockfish suggestions.
Edition Mode: Edit the board, load/save positions, and analyze them.
Piece Detection: Capture a chess.com board screenshot, crop it, and recognize the position using a trained CNN (modelo_piezas.h5).
🚀 Features
GUI built with Pygame.
Stockfish integration for move suggestions (S key = fast suggestion, Shift+S = deeper analysis).
Two board orientations: White or Black perspective.
Edition mode with piece placement tools.
Screenshot & recognition:
Press P in edition mode to capture a region of the screen.
Adjust the crop, press Enter, and the CNN detects the board state automatically.
📦 Installation
Clone this repo:
git clone https://github.com/yourusername/chess-assistant.git
cd chess-assistant
Install dependencies:
pip install pygame python-chess tensorflow pillow numpy
Download and compile Stockfish (if not already available):
git clone https://github.com/official-stockfish/Stockfish.git
cd Stockfish/src
make
Update STOCKFISH_PATH in the code if needed.
▶️ Usage
Play Mode
Run:
python chess_gui_click_and_editor.py --orient white
or
python chess_gui_click_and_editor.py --orient black
Controls:
Click pieces to move.
S → Stockfish best move (1s analysis).
Shift+S → Stockfish deeper move (3s analysis).
U → Undo last move.
Edition Mode
Start the GUI, press E to toggle edition mode.
Place/remove pieces with the mouse.
Press P to capture & detect a chess.com board.
Move the crop box and press Enter to confirm.
The board will auto-populate using modelo_piezas.h5.
🧠 Piece Recognition
The model (modelo_piezas.h5) was trained to classify chess pieces on chess.com style boards.
Input: cropped 224x224 squares.
Output: predicted piece type (P, N, B, R, Q, K, or empty).
You can retrain with your own dataset by running:
python train_model.py
📂 Project Structure
chess-assistant/
│── chess_gui_click_and_editor.py   # Main GUI and engine integration
│── modelo_piezas.h5                # Trained CNN for piece recognition
│── train_model.py                  # Script to retrain the model
│── data/                           # Dataset of chess piece images
│── README.md                       # Project documentation
✅ To-Do
 Improve detection accuracy with more training data.
 Add PGN/FEN import/export.
 Implement drag-and-drop movement.
 Multi-platform packaging (Windows/macOS/Linux).
📜 License
MIT License – free to use and modify.