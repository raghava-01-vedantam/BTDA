import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(predictions)
        probs_flat = probs.view(-1)
        targets_flat = targets.float().view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_coeff


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self._dice = DiceLoss(smooth=smooth)
        self._bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self._dice(predictions, targets)
        bce = self._bce(predictions, targets.float())
        return 0.5 * dice + 0.5 * bce


if __name__ == "__main__":
    preds = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    for criterion in [DiceLoss(), DiceBCELoss()]:
        loss = criterion(preds, targets)
        assert loss.item() < 0.2
    print("Loss checks passed.")
