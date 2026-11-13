import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """0.5 * BCE + 0.5 * (1 - soft Dice), for binary road segmentation.

    logits: Bx1xHxW raw model output (no sigmoid)
    target: BxHxW or Bx1xHxW, values {0, 1}, optionally 255 = ignore
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0,
                 ignore_index=255, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.pos_weight = None if pos_weight is None else torch.tensor([pos_weight])

    def forward(self, logits, target):
        logits = logits.float()                       # float32 even under AMP
        if target.dim() == 3:
            target = target.unsqueeze(1)              # B×H×W → B×1×H×W

        valid = (target != self.ignore_index).float() # 1 = use pixel, 0 = ignore
        target = target.float() * valid               # ignored pixels become 0 (masked anyway)

        # BCE: mean over valid pixels
        pw = None if self.pos_weight is None else self.pos_weight.to(logits.device)
        bce_map = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw,
                                                     reduction="none")
        bce = (bce_map * valid).sum() / valid.sum().clamp_min(1.0)

        # soft Dice over the whole batch
        probs = torch.sigmoid(logits) * valid
        intersection = (probs * target).sum()
        dice = (2 * intersection + self.smooth) / (probs.sum() + target.sum() + self.smooth)
        dice_loss = 1.0 - dice

        loss = self.bce_weight * bce + self.dice_weight * dice_loss
        return loss, {"bce": bce.detach(), "dice_loss": dice_loss.detach()}