import matplotlib.pyplot as plt
import numpy as np
import chess

PIECE_TYPES = [
    (chess.PAWN, chess.WHITE),
    (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE),
    (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK),
    (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK),
    (chess.KING, chess.BLACK),
]

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encodes a python-chess Board object into a (8, 8, 18) tensor.

    Channels:
        [0–5]:  White pieces (P, N, B, R, Q, K)
        [6–11]: Black pieces (P, N, B, R, Q, K)
        [12]:   Side to move (all 1s if white to move, 0s if black)
        [13–16]: Castling rights:
            13: White can castle kingside
            14: White can castle queenside
            15: Black can castle kingside
            16: Black can castle queenside
        [17]:   En passant square (1 at the target square if en passant is legal, else 0)
    """
    board_tensor = np.zeros((8, 8, 18), dtype=np.float32)

    for channel_idx, (piece_type, color) in enumerate(PIECE_TYPES):
        for square in board.pieces(piece_type, color):
            row, col = square_to_tensor_coords(square)
            board_tensor[row, col, channel_idx] = 1.0

    side_to_move_value = 1.0 if board.turn == chess.WHITE else 0.0
    board_tensor[:, :, 12] = side_to_move_value

    board_tensor[:, :, 13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    board_tensor[:, :, 14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    board_tensor[:, :, 15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    board_tensor[:, :, 16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    if board.ep_square is not None:
        row, col = square_to_tensor_coords(board.ep_square)
        board_tensor[row, col, 17] = 1.0

    return board_tensor

def square_to_tensor_coords(square: int) -> tuple[int, int]:
    """
    Converts a python-chess square index (0–63, a1=0, h8=63)
    to (row, col) coordinates suitable for our tensor representation.

    The returned coordinates follow image-like convention:
        (0, 0) = top-left corner = a8
        (7, 7) = bottom-right corner = h1

    Args:
        square (int): Square index from 0 to 63.

    Returns:
        (row, col): Tuple of row and column indices in the tensor.
    """
    row = 7 - (square // 8)  # rank-flip: a8 is row 0
    col = square % 8         # file stays as-is
    return row, col

if __name__ == "__main__":
    board = chess.Board()
    encoded = encode_board(board)

    print("Shape:", encoded.shape)  # (8, 8, 13)
    print("Side to move channel (should be all 1.0):\n", encoded[:, :, 12])

    board.push(chess.Move.from_uci("e2e4"))
    encoded2 = encode_board(board)
    print("After e4, side to move channel (should be all 0.0):\n", encoded2[:, :, 12])