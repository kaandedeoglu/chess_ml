import os
import chess
import torch
import numpy as np
from tqdm import tqdm, trange

from chessml.board_encoding import encode_board
from chessml.move_encoding import move_to_index, TOTAL_MOVE_COUNT
from chessml.train.supervised_model import ChessCNN
from chessml.agents.mcts import run_mcts

def generate_self_play_games(model_path, output_path, num_games=100, num_simulations=100, device="cpu"):
    print(f"Using device: {device}, CUDA available: {torch.cuda.is_available()}")
    model = ChessCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_states = []
    all_policies = []
    all_values = []

    print("Starting..")

    for _ in trange(num_games, desc="Generating Self-Play games"):
        board = chess.Board()
        game_states = []
        game_policies = []

        while not board.is_game_over():
            move_counts = run_mcts(model, board, num_simulations=num_simulations, device=device)
            total_visits = sum(move_counts.values())

            policy = np.zeros(TOTAL_MOVE_COUNT, dtype=np.float32)

            for move, count in move_counts.items():
                policy[move_to_index(move)] = count / total_visits

            game_states.append(encode_board(board))
            game_policies.append(policy)

            moves = list(move_counts.keys())
            visits = np.array([move_counts[m] for m in moves], dtype=np.float32)
            visits /= visits.sum()
            move = np.random.choice(moves, p=visits)
            board.push(move)

            # print(f"Turn {board.fullmove_number}, legal moves: {len(list(board.legal_moves))}")
            print(f"Turn {board.fullmove_number}, legal moves: {len(list(board.legal_moves))}", flush=True)
        
        result = board.result
        if result == "1-0":
            outcome = 1.0
        elif result == "0-1":
            outcome = -1.0
        else:
            outcome = 0.0

        for i, state in enumerate(game_states):
            value = outcome if (i % 2 == 0) else -outcome
            all_states.append(state)
            all_policies.append(game_policies[i])
            all_values.append(value)
        
    X = np.array(all_states, dtype=np.float32)
    y = np.array(all_policies, dtype=np.float32)
    z = np.array(all_values, dtype=np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, z=z)
    print(f"\n💾 Saved self-play dataset to {output_path}")

if __name__ == "__main__":
    generate_self_play_games(
        model_path="checkpoints/chess_cnn_with_value.pth",
        output_path="data/processed/self_play.npz",
        num_games=50,
        num_simulations=100,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

