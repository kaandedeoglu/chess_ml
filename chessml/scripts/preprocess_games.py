import chess.pgn
import os
import numpy as np
from tqdm import tqdm

from chessml.board_encoding import encode_board
from chessml.move_encoding import move_to_index, TOTAL_MOVE_COUNT

def parse_all_pgns(data_dir: str, max_games: int = None):
    """
    Generator that parses all .pgn files in the given directory.
    Yields (board_tensor, move_index, value) tuples.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith(".pgn")]
    total_games = 0

    for file in files:
        path = os.path.join(data_dir, file)
        for sample in parse_pgn_file(path, max_games):
            yield sample
            total_games += 1
            if max_games is not None and total_games >= max_games:
                return
            
def parse_pgn_file(path: str, max_games: int = 1000):
    """
    Parses a PGN file and yields (board_tensor, move_index) for each ply.
    """

    with open(path, 'r', encoding='utf-8') as pgn:
        """
        Yields (board_tensor, move_index) pairs from a PGN file.

        Includes promotion moves. Skips only truly invalid moves.
        """

        games_parsed = 0

        while max_games is None or games_parsed < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            board = game.board()

            # Get final result from PGN header
            result = game.headers.get("Result", "*")
            if result == "1-0":
                final_value = 1.0
            elif result == "0-1":
                final_value = -1.0
            else:
                final_value = 0.0

            for move in game.mainline_moves():
                encoded = encode_board(board)
                move_idx = move_to_index(move)
                value = final_value if board.turn == chess.WHITE else -final_value
                
                yield encoded, move_idx, value
                board.push(move)
            
            games_parsed += 1

def generate_dataset(data_dir: str, output_path: str, max_games: int = None):
    X, y, z = [], [], []

    for board_tensor, move_index, value in tqdm(parse_all_pgns(data_dir, max_games), desc="Parsing PGN"):
        X.append(board_tensor)
        y.append(move_index)
        z.append(value)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    z = np.array(z, dtype=np.float32)

    print(f"✅ Parsed {len(X)} samples")
    print(f"🧠 Input shape: {X.shape}, Label shape: {y.shape}, Value shape: {z.shape}")
    print(f"♟️ Max move index seen: {np.max(y)} / {TOTAL_MOVE_COUNT - 1}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, z=z)
    print(f"💾 Saved dataset to {output_path}")

if __name__ == "__main__":
    generate_dataset(
        data_dir="data",
        output_path="data/processed/chess_dataset.npz",
        max_games=None
    )