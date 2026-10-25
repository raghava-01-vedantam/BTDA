import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )


if __name__ == "__main__":
    model = build_unet()
    model.eval()
    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 1, 256, 256)
    print("U-Net sanity check passed.")
