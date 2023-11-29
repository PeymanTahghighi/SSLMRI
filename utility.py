import torch
import numpy as np
from torch import einsum
import medpy as mp
from skimage.measure import label

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
        bl = einsum("bnwhd,bnwhd->bnwhd", pred, dt);     
        return bl.mean();

def calculate_metric_percase(pred, gt, simple = False):
    s = np.sum(gt);
    dice = mp.dc(pred, gt)
    hd = 0;
    f1 = 0;
    if simple is False:
        if s!=0:
            hd = mp.hd95(pred, gt)
            asd = mp.asd(pred, gt)
            precision = mp.precision(pred, gt)
            se = mp.sensitivity(pred, gt)
            sp = mp.specificity(pred, gt)
            label_gt = label(gt)
            label_gts = np.bincount(label_gt.flat)
            label_pred = label(pred)
            label_preds = np.bincount(label_pred.flat)
            M, N = label_gts.shape[0], label_preds.shape[0]
            index = np.where(label_gts<11)
            if index[0].size !=0:
                for idx in range(index[0].shape[0]):
                    mask = label_gt==index[0][idx]
                    label_gt[mask]=0
                    M=M-1
            index = np.where(label_preds<11)
            if index[0].size !=0:
                for idx in range(index[0].shape[0]):
                    mask = label_pred==index[0][idx]
                    label_pred[mask]=0
                    N=N-1
            H_ij = np.zeros((M, N))
            for i in range(M):
                for j in range(N):
                    H_ij[i, j] = ((label_gt==i) * (label_pred==j)).sum()
            TPg=0
            for i in range(1, M):
                alpha = H_ij[i, 1:].sum() / (H_ij[i, :].sum() + 1e-18)
                if alpha > 0.1:
                    wsum, k, vaccept=0, 0, True
                    while wsum < 0.65:
                        pk = np.argsort(-H_ij[i, 1:])[k]+1#np.argwhere(np.argsort(H_ij[i])==k)[0][0]
                        tk = H_ij[0, pk] / H_ij[:, pk].sum()
                        if tk >0.7:
                            vaccept = False
                            break
                        wsum += H_ij[i, pk] / H_ij[i, 1:].sum()
                        k +=1
                    if vaccept == True:
                        TPg +=1
            TPa=0
            H_ji = H_ij.T
            for j in range(1, N):
                alpha = H_ji[j, 1:].sum() / (H_ji[j, :].sum()+ 1e-18)
                if alpha > 0.1:
                    wsum, k, vaccept=0, 0, True
                    while wsum < 0.65:
                        pk = np.argsort(-H_ji[j, 1:])[k]+1#np.argwhere(np.argsort(H_ji[j])==k)[0][0]
                        tk = H_ji[0, pk] / H_ji[:, pk].sum()
                        if tk >0.7:
                            vaccept = False
                            break
                        wsum += H_ji[j, pk] / H_ji[j, 1:].sum()
                        k +=1
                    if vaccept == True:
                        TPa +=1
            sel, pl = TPg/((M-1)+1e-6),TPa/((N-1)+1e-6)
            if sel == 0 or pl == 0:
                f1 = 0
            else:
                f1 = (2 * sel * pl) / (sel+pl)
            # print("TPg:{}, M:{}, TPa:{}, N:{}".format(TPg, M-1, TPa, N-1))
            # print("sel:{}, pl:{}, f1:{}".format(sel, pl, F1))
            # print("dice:{}, jc:{}, 95hd:{}, asd:{}, pr:{}, se:{}, sp:{}".format(dice, jc, hd, asd, precision, se, sp))
        return dice, hd, f1;
    else:
        return dice;