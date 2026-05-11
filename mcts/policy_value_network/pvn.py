import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return F.relu(out)


class PVN(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super(PVN, self).__init__()

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.middle_blocks = nn.Sequential(
            *[ResNetBlock(out_channels) for _ in range(5)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(out_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, 81)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 9 * 9, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.init_layer(x)
        x = self.middle_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value