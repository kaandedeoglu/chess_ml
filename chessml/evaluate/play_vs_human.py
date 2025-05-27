import torch
import chess
import chess.engine
from chessml.train.supervised_model import ChessCNN
from chessml.board_encoding import encode_board
from chessml.move_encoding import move_to_index, index_to_move

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def pick_model_move(model, board):
    x = encode_board(board)
    x_tensor = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 18, 8, 8]
    
    with torch.no_grad():
        logits, _ = model(x_tensor)

    # Mask out illegal moves
    mask = torch.full_like(logits, float("-inf"))
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = logits[idx]

    # Pick the legal move with highest logit
    move_idx = mask.argmax().item()
    predicted_move = index_to_move(move_idx)

    if predicted_move not in board.legal_moves:
        print("⚠️  Model predicted illegal move. Picking random legal move instead.")
        return next(iter(board.legal_moves))

    return predicted_move

def main():
    print("♟️  Welcome to ChessML! Play against your trained model.")
    print("Do you want to play as white or black? [w/b]")
    color = input(">>> ").strip().lower()
    human_white = color in ["w", "white", ""]

    board = chess.Board()

    # Load model
    model = ChessCNN().to(device)
    model.load_state_dict(torch.load("checkpoints/chess_cnn_with_value.pth", map_location=device))
    model.eval()
    print("🧠 Model loaded.")

    while not board.is_game_over():
        print("\n" + str(board) + "\n")

        if board.turn == chess.WHITE and human_white or board.turn == chess.BLACK and not human_white:
            move_str = input("Your move (UCI format, e.g. e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    raise ValueError
                board.push(move)
            except:
                print("Invalid move. Try again.")
        else:
            print("🤖 Model is thinking...")
            move = pick_model_move(model, board)
            print(f"Model plays: {move.uci()}")
            board.push(move)

    print("\n🏁 Game over!")
    print("Result:", board.result())
    print(board.outcome().termination.name)

if __name__ == "__main__":
    main()