"""
Basic CNN model for image classification.

TinyVGG architecture - based on the one from CNN Explainer:
https://poloclub.github.io/cnn-explainer/

It's just 2 conv blocks (conv -> relu -> conv -> relu -> maxpool)
followed by a linear layer that spits out the class scores.
"""
import torch
from torch import nn


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        """
        input_shape:  number of color channels in the input image (3 for RGB)
        hidden_units: number of filters used in each conv layer
        output_shape: number of classes to predict
        """
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # halves height & width
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # halves height & width again
        )

        # image shrinks 64x64 -> 32x32 -> 16x16 after the two blocks above,
        # so the flattened size is hidden_units * 16 * 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 16 * 16, output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
