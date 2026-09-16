import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )


class TinyCNN(nn.Module):
    """Compact CNN baseline for CIFAR-10 and later INT8 experiments."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32),
            ConvBNReLU(32, 32),
            nn.MaxPool2d(2),
            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Build one of the two architectures defined by the project methodology."""
    key = name.lower().replace("-", "").replace("_", "")

    if key == "tinycnn":
        return TinyCNN(num_classes=num_classes)

    if key == "mobilenetv2":
        # Train from scratch on CIFAR-10. Keep the standard MobileNetV2 body
        # unchanged so later deployment/quantization comparisons stay clean.
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model

    raise ValueError(
        f"Unknown model '{name}'. Supported models: tinycnn, mobilenetv2."
    )


def canonical_model_name(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "")
    if key == "tinycnn":
        return "tinycnn"
    if key == "mobilenetv2":
        return "mobilenetv2"
    raise ValueError(
        f"Unknown model '{name}'. Supported models: tinycnn, mobilenetv2."
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
