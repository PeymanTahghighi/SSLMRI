import torch
import numpy as np
from torch import einsum

def IoU(mri1, mri2):
    dims = [d for d in range(1, mri1.ndim)];
    intersection = torch.sum(mri1 * mri2, dim = dims);
    union = torch.sum(mri1 + mri2, dim = dims);
    return torch.mean(intersection / (union+1e-4));

class BounraryLoss(object):
    def __init__(self, sigmoid = True) -> None:
        self.sigmoid = sigmoid;
    def __call__(self, pred, dt):
        if self.sigmoid:
            pred = torch.sigmoid(pred);
        axis = [i for i in range(2, pred.ndim)];
        bl = einsum("bnwhd,bnwhd->bnwhd", pred, dt);
        
        return bl.mean();

