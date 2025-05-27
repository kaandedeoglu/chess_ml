import matplotlib.pyplot as plt
import numpy as np

CHANNEL_NAMES = [
    "White P", "White N", "White B", "White R", "White Q", "White K",
    "Black P", "Black N", "Black B", "Black R", "Black Q", "Black K",
    "To Move (White=1)",
    "W K-side", "W Q-side", "B K-side", "B Q-side"
]

def visualize_board_tensor(tensor: np.ndarray):
    """
    Visualize all channels of the board tensor using heatmaps.
    """
    num_channels = tensor.shape[-1]
    fig, axes = plt.subplots(3, 6, figsize=(15, 8)) # 18 slots in a 3x6 grid

    for i in range(num_channels):
        ax = axes[i // 6, i % 6]
        ax.imshow(tensor[:, :, i], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"Channel {i}")
        ax.axis("off")
    
    for j in range(num_channels, 18):
        fig.delaxes(axes[j // 6, j % 6])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import chess
    from chessml.board_encoding import encode_board

    board = chess.Board()
    tensor = encode_board(board)

    from chessml.visualize import visualize_board_tensor
    visualize_board_tensor(tensor)