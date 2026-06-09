import torch
import torch.nn as nn

class RankMSELoss(nn.Module):
    def __init__(self, alpha=0.5, margin=0.1):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        mse = self.mse(preds, targets)
        pred_diff = preds - preds.t()
        target_diff = targets - targets.t()
        S = torch.sign(target_diff)
        rank_loss = torch.relu(-S * pred_diff + self.margin)
        mask = (S != 0).float()
        rank_loss = (rank_loss * mask).sum() / (mask.sum() + 1e-8)
        return self.alpha * mse + (1 - self.alpha) * rank_loss
