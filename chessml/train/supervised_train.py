import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from chessml.train.supervised_model import ChessCNN
from chessml.train.supervised_dataset import ChessMoveDataset

def train_model(
    npz_path="data/processed/chess_dataset.npz",
    batch_size=64,
    epochs=5,
    lr=1e-3,
    device=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    # Dataset + DataLoader
    dataset = ChessMoveDataset(npz_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Model, loss, optimizer
    model = ChessCNN().to(device)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        correct = 0

        for X_batch, y_batch, z_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            z_batch = z_batch.to(device)

            optimizer.zero_grad()
            policy_logits, predicted_value = model(X_batch)

            policy_loss = criterion_policy(policy_logits, y_batch)
            value_loss = criterion_value(predicted_value, z_batch)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            correct += (policy_logits.argmax(dim=1) == y_batch).sum().item()

        accuracy = correct / len(dataset) * 100
        print(f"📦 Epoch {epoch+1}: Total Loss = {total_loss/len(loader):.4f}, "
              f"Policy Loss = {total_policy_loss/len(loader):.4f}, "
              f"Value Loss = {total_value_loss/len(loader):.4f}, "
              f"Accuracy = {accuracy:.2f}%")

    torch.save(model.state_dict(), "checkpoints/chess_cnn_with_value.pth")
    print("💾 Model saved to checkpoints/chess_cnn_with_value.pth")
    return model

if __name__ == "__main__":
    train_model(
        npz_path="data/processed/chess_dataset.npz",
        batch_size=256,
        epochs=120,
        lr=1e-3,
    )