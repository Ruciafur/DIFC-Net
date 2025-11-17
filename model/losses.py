# losses.py
import torch
import torch.nn as nn

class LabelSmoothingBCELoss(nn.Module):
    """
    对二分类 BCEWithLogitsLoss 做简单标签平滑:
      y_smooth = y*(1-ε) + 0.5*ε
    """
    def __init__(self, eps: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets):
        # targets: float tensor in {0,1}
        with torch.no_grad():
            targets = targets * (1.0 - self.eps) + 0.5 * self.eps
        return self.bce(logits, targets)
