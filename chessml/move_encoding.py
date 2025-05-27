import chess

NUM_STANDARD_MOVES = 64 * 64
PROMOTION_OFFSET = NUM_STANDARD_MOVES
NUM_PROMOTION_TYPES = 4
TOTAL_MOVE_COUNT = NUM_STANDARD_MOVES + (NUM_STANDARD_MOVES * NUM_PROMOTION_TYPES)

PROMOTION_PIECE_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

def move_to_index(move: chess.Move) -> int:
    """
    Encodes a chess.Move into a unique integer in [0, TOTAL_MOVE_COUNT).
    
    Encoding logic:
    - For standard moves (no promotion):
        index = from_square * 64 + to_square
    - For promotion moves:
        index = PROMOTION_OFFSET + (from_square * 64 + to_square) * 4 + promotion_type_index

    This ensures:
        - All 4096 standard moves occupy [0, 4095]
        - All promotions occupy [4096, 20479]
        - Index space is dense, unique, and reversible
    """
    if move.promotion is None:
        return move.from_square * 64 + move.to_square
    else:
        promo_type_idx = PROMOTION_PIECE_TYPES.index(move.promotion)
        base_index = move.from_square * 64 + move.to_square
        return PROMOTION_OFFSET + base_index * NUM_PROMOTION_TYPES + promo_type_idx

def index_to_move(index: int) -> chess.Move:
    """
    Decodes a unique move index back into a chess.Move.

    Decoding logic:
    - If index < 4096: it's a standard move
        → from = index // 64, to = index % 64
    - If index >= 4096: it's a promotion
        → base = (index - PROMOTION_OFFSET) // 4
          promo_type_idx = (index - PROMOTION_OFFSET) % 4
          → from = base // 64, to = base % 64
          → promotion = PROMOTION_PIECE_TYPES[promo_type_idx]
    """
    if index < PROMOTION_OFFSET:
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)
    else :
        promo_index = index - PROMOTION_OFFSET
        base = promo_index // NUM_PROMOTION_TYPES
        prompo_type_idx = promo_index % NUM_PROMOTION_TYPES
        from_square = base // 64
        to_square = base % 64
        promotion = PROMOTION_PIECE_TYPES[prompo_type_idx]
        return chess.Move(from_square, to_square, promotion=promotion)
    
def move_index_to_uci_string(index: int) -> str:
    """
    Converts a move index back into a UCI move string (e.g., 'e2e4', 'e7e8q').

    Useful for logging and debugging model predictions.
    """
    move = index_to_move(index)
    return move.uci()

if __name__ == "__main__":
    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(move)
    print("Index:", idx)

    recovered = index_to_move(idx)
    print("Recovered move:", recovered.uci())