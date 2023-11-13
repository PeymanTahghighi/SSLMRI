import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu';
hyperparameters = {
    'batch_size' : 4,
    'crop_size_w': 96,
    'crop_size_h': 96,
    'crop_size_d':96,
    'learning_rate':1e-4,
    'sample_per_mri':8,
    'deterministic':False,
    'virtual_batch_size':1,
    'num_workers': 8,
    'bl_multiplier': 20
}
DEBUG_TRAIN_DATA = False;
PRETRAINED = False
PRERTRAIN_PATH = 'exp/pretrain-miccai/best_model.ckpt';