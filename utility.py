import torch
import numpy as np

def IoU(mri1, mri2):
    dims = [d for d in range(1, mri1.ndim)];
    intersection = torch.sum(mri1 * mri2, dim = dims);
    union = torch.sum(mri1 + mri2, dim = dims);
    return torch.mean(intersection / (union+1e-4));

class MimeLoss(object):
    def __init__(self, a, b, sigmoid = True) -> None:
        self.a = a;
        self.b = b;
        self.sigmoid = sigmoid;
    def __call__(self, gt, pred):
        if self.sigmoid:
            pred = torch.sigmoid(pred);
        gt = -gt*self.a + (1-gt)*self.b;
        return torch.mean(gt*pred);

