import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessCNN(nn.Module):
    def __init__(self, num_classes=20480):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.shared(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value.squeeze(-1)