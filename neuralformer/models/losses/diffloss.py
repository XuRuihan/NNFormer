import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiffLoss(nn.Module):
    def __init__(self, type: str):
        super().__init__()
        if type.lower() == "l1":
            self.cal_loss = nn.L1Loss()
        elif type.lower() == "l2":
            self.cal_loss = nn.MSELoss()
        elif type.lower() == "kldiv":
            self.cal_loss = nn.KLDivLoss()

    def forward(self, predicts: Tensor, target: Tensor) -> Tensor:
        index = torch.randperm(predicts.shape[0], device=predicts.device)
        v1 = predicts - predicts[index]
        v2 = target - target[index]
        loss = self.cal_loss(v1, v2)
        return loss

    # def forward(self, predicts: Tensor, target: Tensor):
    #     v1 = predicts.unsqueeze(1) - predicts.unsqueeze(0)
    #     v2 = target.unsqueeze(1) - target.unsqueeze(0)
    #     loss = 500 * F.mse_loss(v1, v2)
    #     return loss


if __name__ == "__main__":
    loss = DiffLoss("l1")
    a1 = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=float)
    a2 = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], dtype=float)
    assert loss(a1, a2).item() < 1e-5

    b1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    b2 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    assert loss(b1, b2).item() < 1e-5
