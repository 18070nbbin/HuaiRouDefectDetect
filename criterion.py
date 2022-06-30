import torch
from PIL import Image

def focal_loss(x, y, alpha=50, gamma=2):
    """
    Use focal_loss to give positive sample more weight.
    """
    N, C, H, D = x.shape
    x = torch.sigmoid(x)
    x = x.clamp(min=1e-5, max=1 - 1e-5)# prevent loss->nan
    x = x.reshape(-1)
    y = y.reshape(-1)
    positive = (y == 1)
    negative = (y == 0)

    p = x[positive]
    positive_loss = torch.sum(- (1 - p) ** gamma * torch.log(p))
    n = x[negative]
    negative_loss = torch.sum(-n ** gamma * torch.log(1 - n))

    return (negative_loss + alpha * positive_loss) / (N * C * H * D)
