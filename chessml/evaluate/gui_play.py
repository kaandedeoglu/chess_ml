import pygame
import chess
import torch
from chessml.train.supervised_model import ChessCNN
from chessml.board_encoding import encode_board
from chessml.move_encoding import move_to_index, index_to_move
from chessml.agents.mcts import run_mcts
import cairosvg
import io

def load_svg_as_surface(path, size):
    png_bytes = cairosvg.svg2png(url=path, output_width=size, output_height=size)
    image_io = io.BytesIO(png_bytes)
    return pygame.image.load(image_io, 'image.png').convert_alpha()

# Config
SQUARE_SIZE = 80
WIDTH, HEIGHT = 8 * SQUARE_SIZE, 8 * SQUARE_SIZE
FPS = 30
LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)

PIECE_IMAGES = {}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def draw_board(screen, board):
    colors = [LIGHT_COLOR, DARK_COLOR]
    for rank in range(8):
        for file in range(8):
            color = colors[(rank + file) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            screen.blit(PIECE_IMAGES[piece.symbol()], (col * SQUARE_SIZE, row * SQUARE_SIZE))

def square_from_mouse(pos):
    x, y = pos
    col = x // SQUARE_SIZE
    row = 7 - (y // SQUARE_SIZE)
    return chess.square(col, row)

def pick_model_move(model, board):
    x = encode_board(board)
    x_tensor = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(x_tensor)
        logits = logits.squeeze(0)

    mask = torch.full_like(logits, float("-inf"))
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = logits[idx]

    k = 5  # 🧪 try different values: 3, 5, 10
    topk = torch.topk(mask, k=k)
    topk_indices = topk.indices
    topk_logits = topk.values

    # Turn logits into probabilities
    probs = torch.softmax(topk_logits, dim=0)
    sampled_idx = torch.multinomial(probs, 1).item()
    move_idx = topk_indices[sampled_idx].item()
    move = index_to_move(move_idx)

    return move if move in board.legal_moves else next(iter(board.legal_moves))

def pick_move_with_mcts(model, board, num_simulations=100, temperature=1.0):
    device = next(model.parameters()).device
    move_counts = run_mcts(model, board, num_simulations=num_simulations, device=device)

    if temperature == 0:
        return max(move_counts, key=move_counts.get)
    
    moves = list(move_counts.keys())
    visits = torch.tensor([move_counts[m] for m in moves], dtype=torch.float32)
    probs = (visits ** (1.0 / temperature))
    probs /= probs.sum()

    sampled_idx = torch.multinomial(probs, 1).item()
    return moves[sampled_idx]

def load_piece_images():
    pieces = "prnbqkPRNBQK"
    for p in pieces:
        path = f"chessml/evaluate/assets_svg/{p}.svg"
        PIECE_IMAGES[p] = load_svg_as_surface(path, SQUARE_SIZE)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ChessML GUI")
    clock = pygame.time.Clock()

    board = chess.Board()
    selected_square = None

    # Load model
    model = ChessCNN().to(device)
    model.load_state_dict(torch.load("checkpoints/chess_cnn_with_value.pth", map_location=device))
    model.eval()

    load_piece_images()
    running = True

    while running:
        screen.fill((0, 0, 0))
        draw_board(screen, board)
        pygame.display.flip()
        clock.tick(FPS)

        if board.is_game_over():
            print("Game over:", board.result())
            pygame.time.wait(3000)
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and board.turn == chess.WHITE:
                clicked = square_from_mouse(event.pos)
                if selected_square is None:
                    if board.piece_at(clicked) and board.piece_at(clicked).color == chess.WHITE:
                        selected_square = clicked
                else:
                    move = chess.Move(selected_square, clicked)
                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None
                    else:
                        selected_square = None

        if board.turn == chess.BLACK and not board.is_game_over():
            # move = pick_model_move(model, board)
            move = pick_move_with_mcts(model, board, num_simulations=400, temperature=0.5)
            board.push(move)

    pygame.quit()

if __name__ == "__main__":
    main()