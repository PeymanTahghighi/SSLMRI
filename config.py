import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu';
hyperparameters = {
    'batch_size' : 2,
    'crop_size_w': 64,
    'crop_size_h': 128,
    'crop_size_d':128,
    'learning_rate':1e-4,
    'sample_per_mri':4,
    'deterministic':False,
    'virtual_batch_size':1
}