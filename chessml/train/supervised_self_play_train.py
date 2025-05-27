import os
import chess
import torch
import numpy as np
from tqdm import tqdm, trange

from chessml.board_encoding import encode_board
from chessml.move_encoding import move_to_index, TOTAL_MOVE_COUNT
from chessml.train.supervised_model import ChessCNN
from chessml.agents.mcts import run_mcts
from chessml.train.supervised_dataset import ChessMoveDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def generate_self_play_games(
    model_path,
    output_path,
    num_games=1000,
    num_simulations=100,
    device="cpu",
    verbose=False
):
    model = ChessCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_states = []
    all_policies = []
    all_values = []

    for game_idx in trange(num_games, desc="Generating Self-Play games"):
        board = chess.Board()
        game_states = []
        game_policies = []

        while not board.is_game_over():
            move_counts = run_mcts(model, board, num_simulations=num_simulations, device=device)
            total_visits = sum(move_counts.values())

            tqdm.write(f"Game {game_idx}, Turn {board.fullmove_number}, legal moves: {len(list(board.legal_moves))}")

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

        result = board.result()
        outcome = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0

        for i, state in enumerate(game_states):
            value = outcome if (i % 2 == 0) else -outcome
            all_states.append(state)
            all_policies.append(game_policies[i])
            all_values.append(value)

        if verbose and game_idx % 50 == 0:
            print(f"Game {game_idx} result: {result}")

    X = np.array(all_states, dtype=np.float32)
    y = np.array(all_policies, dtype=np.float32)
    z = np.array(all_values, dtype=np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, z=z)
    print(f"\n💾 Saved self-play dataset to {output_path}")

def train_model(npz_path="data/processed/self_play_large.npz", batch_size=256, epochs=10, lr=1e-3, device="cuda"):
    print(f"🧠 Training on device: {device}")

    dataset = ChessMoveDataset(npz_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = ChessCNN().to(device)
    criterion_policy = nn.KLDivLoss()
    criterion_value = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for X_batch, y_batch, z_batch in loader:
            X_batch, y_batch, z_batch = X_batch.to(device), y_batch.to(device), z_batch.to(device)

            optimizer.zero_grad()
            policy_logits, predicted_value = model(X_batch)

            policy_log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = criterion_policy(policy_log_probs, y_batch)
            value_loss = criterion_value(predicted_value.squeeze(-1), z_batch)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (policy_logits.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()

        print(f"📦 Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Accuracy={correct/len(dataset)*100:.2f}%")

    torch.save(model.state_dict(), "checkpoints/chess_cnn_with_value.pth")
    print("💾 Model saved to checkpoints/chess_cnn_with_value.pth")
    return model

if __name__ == "__main__":
    generate_self_play_games(
        model_path="checkpoints/chess_cnn_with_value.pth",
        output_path="data/processed/self_play_large.npz",
        num_games=1000,  # 🔁 scale this up to 5000+ as needed
        num_simulations=100,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=True
    )
