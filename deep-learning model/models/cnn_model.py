"""
CNN Model Definition for Habitat Classification

This module contains the ResNet18-based model used for habitat classification.
"""

import torch
import torch.nn as nn
from torchvision import models


class PretrainedResNet18(nn.Module):
    """
    ResNet18 model with custom input layer and classifier head.
    Modified to accept multi-channel medical images.
    """

    def __init__(self, in_chans=15, num_classes=2, pretrained=True):
        super().__init__()

        # Load pre-trained ResNet18
        backbone = models.resnet18(weights='DEFAULT' if pretrained else None)

        # Modify first conv layer to accept custom number of input channels
        original_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_chans,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )

        # Initialize weights by averaging across input channels
        if pretrained:
            with torch.no_grad():
                pretrained_weight = original_conv1.weight.data
                avg_weight = pretrained_weight.mean(dim=1, keepdim=True)
                new_weight = avg_weight.repeat(1, in_chans, 1, 1)
                new_conv1.weight.data = new_weight

        backbone.conv1 = new_conv1

        # Replace classification head with dropout for regularization
        num_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def create_model(config):
    """
    Create the ResNet18 model based on configuration.

    Args:
        config: Configuration object containing IN_CHANNELS and NUM_CLASSES

    Returns:
        PretrainedResNet18 model instance
    """
    model = PretrainedResNet18(
        in_chans=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        pretrained=True
    )
    print(f"✅ Using ImageNet pre-trained ResNet18")
    print(f"   Input channels: {config.IN_CHANNELS}, Number of classes: {config.NUM_CLASSES}")
    return model


# Test the model
if __name__ == "__main__":
    class MockConfig:
        IN_CHANNELS = 15
        NUM_CLASSES = 2

    model = create_model(MockConfig())
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    x = torch.randn(2, 15, 128, 128)
    output = model(x)
    print(f"Input: {x.shape}, Output: {output.shape}")