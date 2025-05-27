import numpy as np
import torch
from torch.utils.data import Dataset

class ChessMoveDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = data["X"]
        self.y = data["y"]
        self.z = data["z"]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        Torch expects data to be arranged as [C, H, W]
        So permute our preprocessed data that's arranged as [H, W, C] to match this shape.
        """
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32).permute(2, 0, 1)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        z_tensor = torch.tensor(self.z[idx], dtype=torch.float32)
        return x_tensor, y_tensor, z_tensor