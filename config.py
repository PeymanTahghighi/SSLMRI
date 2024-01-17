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
    'num_workers': 0,
    'bl_multiplier': 5
}

DEBUG_TRAIN_DATA = False;
PRETRAINED = False;
USE_ONE_SAMPLE = False;
FOLD = 2;
PRERTRAIN_PATH = f'exp/Pretraining MICCAI16/best_model.ckpt';