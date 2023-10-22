from data_utils import cache_mri_gradients, get_loader, cache_test_dataset, update_folds, update_folds_isbi, get_loader_isbi, visualize_2d, update_folds_miccai, get_loader_miccai
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch
import config
import torch.nn.functional as F
from model_3d import UNet3D, CrossAttentionUNet3D, ResUnet3D
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision.utils import save_image, make_grid
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
from sklearn.metrics import precision_recall_fscore_support
from monai.losses.dice import DiceLoss, DiceFocalLoss
from skimage.filters import threshold_otsu
import seaborn as sns
#===============================================================
def dice_loss(input, 
                target, 
                eps=1e-7, 
                sigmoid = False,
                multilabel = False):

    if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {}"
                            .format(input.shape, input.shape))
    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}" .format(
                input.device, target.device))
    # compute softmax over the classes axis
    #input_soft = torch.sigmoid(input)
    if sigmoid is True:
        target = torch.sigmoid(target);


    # compute the actual dice score
    dims = (1, 2, 3, 4)
    intersection = torch.sum(input * target, dims)
    cardinality = torch.sum(input + target, dims)

    dice_score = 2. * intersection / (cardinality + 1e-4)
    return torch.mean(1. - dice_score)
#===============================================================

def train(model, train_loader, optimizer, scalar):
    print(('\n' + '%10s'*3) %('Epoch', 'Loss', 'IoU'));
    pbar = tqdm(enumerate(train_loader), total= len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epoch_loss = [];
    epoch_IoU = [];
    curr_step = 0;
    curr_iou = 0;
    for batch_idx, (batch) in pbar:
        mri, mri_noisy, heatmap = batch[0].to('cuda').unsqueeze(dim=1), batch[1].to('cuda').unsqueeze(dim=1), batch[2].to('cuda')
        steps = config.hyperparameters['sample_per_mri'] // config.hyperparameters['batch_size'];
        curr_loss = 0;
        for s in range(steps):
            curr_mri = mri[s*config.hyperparameters['batch_size']:(s+1)*config.hyperparameters['batch_size']]
            curr_mri_noisy = mri_noisy[s*config.hyperparameters['batch_size']:(s+1)*config.hyperparameters['batch_size']]
            curr_heatmap = heatmap[s*config.hyperparameters['batch_size']:(s+1)*config.hyperparameters['batch_size']]

            assert not torch.any(torch.isnan(curr_mri)) or not torch.any(torch.isnan(curr_mri_noisy)) or not torch.any(torch.isnan(curr_heatmap))
            with torch.cuda.amp.autocast_mode.autocast():
                hm1 = model(curr_mri, curr_mri_noisy);
                hm2 = model(curr_mri_noisy, curr_mri);
                lih1 = F.l1_loss((curr_mri+hm1), curr_mri_noisy);
                lih2 = F.l1_loss((curr_mri_noisy+hm2), curr_mri);
                lhh = F.l1_loss((hm1+hm2), torch.zeros_like(hm1));
                lh1 = F.l1_loss((hm1)*curr_heatmap, torch.zeros_like(hm1));
                lh2 = F.l1_loss((hm2)*curr_heatmap, torch.zeros_like(hm1));
                loss = (lih1 + lih2 + lhh + lh1 + lh2)/ config.hyperparameters['virtual_batch_size'];

            scalar.scale(loss).backward();
            curr_loss += loss.item();
            curr_step+=1;


            if (curr_step) % config.hyperparameters['virtual_batch_size'] == 0:
                scalar.step(optimizer);
                scalar.update();
                
                model.zero_grad(set_to_none = True);
                epoch_loss.append(curr_loss);
                epoch_IoU.append(curr_iou);
                curr_loss = 0;
                curr_step = 0;
                curr_iou = 0;

            pbar.set_description(('%10s' + '%10.4g'*2)%(epoch, np.mean(epoch_loss)));

    return np.mean(epoch_loss);


def valid(model, loader):
    print(('\n' + '%10s'*2) %('Epoch', 'Loss'));
    pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epoch_loss = [];
    with torch.no_grad():
        for idx, (batch) in pbar:
            mri, mri_noisy, heatmap = batch[0].to('cuda').unsqueeze(dim=1), batch[1].to('cuda').unsqueeze(dim=1), batch[2].to('cuda')
            hm1 = model(mri, mri_noisy);
            hm2 = model(mri_noisy, mri);
            lih1 = F.l1_loss((mri+hm1), mri_noisy);
            lih2 = F.l1_loss((mri_noisy+hm2), mri);
            lhh = F.l1_loss((hm1+hm2), torch.zeros_like(hm1));
            lh1 = F.l1_loss((hm1)*heatmap, torch.zeros_like(hm1));
            lh2 = F.l1_loss((hm2)*heatmap, torch.zeros_like(hm1));
            total_loss = lih1 + lih2 + lhh + lh1 + lh2;


            epoch_loss.append(total_loss.item());
            pbar.set_description(('%10s' + '%10.4g')%(epoch, np.mean(epoch_loss)));

    return np.mean(epoch_loss);

def save_examples_miccai(model, loader,):
    if os.path.exists('samples') is False:
        os.makedirs('samples');
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        counter = 0;
        for idx, (batch) in pbar:
            mri, mri_noisy, heatmap, lbl, brainmask = batch[0].to('cuda'), batch[1].to('cuda'), batch[2].to('cuda'), batch[3], batch[4].to('cuda');
            if torch.sum(heatmap).item() > 0:
                hm1 = model(mri, mri_noisy);
                hm2 = model(mri_noisy, mri);
                heatmap = heatmap.detach().cpu().numpy();
                pred_lbl_1 = torch.sigmoid(hm1)>0.5;
                pred_lbl_2 = torch.sigmoid(hm2)>0.5;
                pred = (pred_lbl_1 * pred_lbl_2*brainmask).detach().cpu().numpy();
                hm1 = hm1.detach().cpu().numpy();
                hm2 = hm2.detach().cpu().numpy();
                mri = mri.detach().cpu().numpy();
                mri_noisy = mri_noisy.detach().cpu().numpy();

                for j in range(2):
                    #heatmap = (1-heatmap) > 0;
                    pos_cords = np.where(heatmap[0] >0);
                    if len(pos_cords[0]) != 0:
                        r = np.random.randint(0,len(pos_cords[0]));
                        center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
                    else:
                        center = [hm1.shape[2]//2, hm1.shape[3]//2, hm1.shape[4]//2]
                    fig, ax = plt.subplots(2,6);
                    ax[0][0].imshow(pred[0,0,center[0], :, :], cmap='hot');
                    ax[0][1].imshow(pred[0,0,:,center[1], :], cmap='hot');
                    ax[0][2].imshow(pred[0,0, :, :,center[2]], cmap='hot');

                    ax[0][3].imshow(heatmap[0,0,center[0], :, :]);
                    ax[0][4].imshow(heatmap[0,0,:,center[1], :]);
                    ax[0][5].imshow(heatmap[0,0, :, :,center[2]]);

                    ax[1][0].imshow(mri[0,0,center[0], :, :], cmap='gray');
                    ax[1][1].imshow(mri[0,0,:,center[1], :], cmap='gray');
                    ax[1][2].imshow(mri[0,0, :, :,center[2]], cmap='gray');

                    ax[1][3].imshow(mri_noisy[0,0,center[0], :, :], cmap='gray');
                    ax[1][4].imshow(mri_noisy[0,0,:,center[1], :], cmap='gray');
                    ax[1][5].imshow(mri_noisy[0,0, :, :,center[2]], cmap='gray');

                    fig.savefig(os.path.join('samples',f'sample_{epoch}_{counter + idx*config.hyperparameters["batch_size"]}_{j}.png'));
                    plt.close("all");

                counter += 1;
                if counter >=5:
                    break;

def save_examples(model, loader,):
    if os.path.exists('samples') is False:
        os.makedirs('samples');
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        counter = 0;
        for idx, (batch) in pbar:
            mri, mri_noisy, heatmap, lbl = batch[0].to('cuda'), batch[1].to('cuda'), batch[2].to('cuda'), batch[3].to('cuda');
            hm1 = model(mri, mri_noisy);
            hm2 = model(mri_noisy, mri);            
            mri_recon = (mri_noisy+hm2);
            mri_noisy_recon = (mri+hm1);

            heatmap = heatmap.detach().cpu().numpy();
            hm1 = hm1.detach().cpu().numpy();
            hm2 = hm2.detach().cpu().numpy();
            mri_recon = mri_recon.detach().cpu().numpy();
            mri_noisy_recon = mri_noisy_recon.detach().cpu().numpy();
            mri = mri.detach().cpu().numpy();
            mri_noisy = mri_noisy.detach().cpu().numpy();
            for j in range(2):
                pos_cords = np.where(heatmap[0] >0);
                if len(pos_cords[0]) != 0:
                    r = np.random.randint(0,len(pos_cords[0]));
                    center = [pos_cords[0][r], pos_cords[1][r],pos_cords[2][r]]
                else:
                    center = [hm1.shape[2]//2, hm1.shape[3]//2, hm1.shape[4]//2]
                fig, ax = plt.subplots(3,6);
                ax[0][0].imshow(hm1[0,0,center[0], :, :], cmap='hot');
                ax[0][1].imshow(hm1[0,0,:,center[1], :], cmap='hot');
                ax[0][2].imshow(hm1[0,0, :, :,center[2]], cmap='hot');

                ax[0][3].imshow(hm2[0,0,center[0], :, :], cmap='hot');
                ax[0][4].imshow(hm2[0,0,:,center[1], :], cmap='hot');
                ax[0][5].imshow(hm2[0,0, :, :,center[2]], cmap='hot');

                ax[1][0].imshow(mri[0,0,center[0], :, :], cmap='gray');
                ax[1][1].imshow(mri[0,0,:,center[1], :], cmap='gray');
                ax[1][2].imshow(mri[0,0, :, :,center[2]], cmap='gray');

                ax[1][3].imshow(mri_noisy[0,0,center[0], :, :], cmap='gray');
                ax[1][4].imshow(mri_noisy[0,0,:,center[1], :], cmap='gray');
                ax[1][5].imshow(mri_noisy[0,0, :, :,center[2]], cmap='gray');

                ax[2][0].imshow(mri_noisy_recon[0,0,center[0], :, :], cmap='gray');
                ax[2][1].imshow(mri_noisy_recon[0,0,:,center[1], :], cmap='gray');
                ax[2][2].imshow(mri_noisy_recon[0,0, :, :,center[2]], cmap='gray');

                ax[2][3].imshow(mri_recon[0,0,center[0], :, :], cmap='gray');
                ax[2][4].imshow(mri_recon[0,0,:,center[1], :], cmap='gray');
                ax[2][5].imshow(mri_recon[0,0, :, :,center[2]], cmap='gray');
                fig.savefig(os.path.join('samples',f'sample_{epoch}_{counter + idx*config.hyperparameters["batch_size"]}_{j}.png'));
                plt.close("all");

            counter += 1;
            if counter >=5:
                break;

def train_miccai(model, train_loader, optimizer, scalar):
    print(('\n' + '%10s'*3) %('Epoch', 'Loss', 'IoU'));
    pbar = tqdm(enumerate(train_loader), total= len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epoch_loss = [];
    epoch_IoU = [];
    curr_step = 0;
    curr_iou = 0;
    for batch_idx, (batch) in pbar:
        mri, mri_noisy, heatmap = batch[0].to('cuda').squeeze().unsqueeze(dim=1), batch[1].to('cuda').squeeze().unsqueeze(dim=1), batch[2].to('cuda').squeeze(dim=0)
        steps = config.hyperparameters['sample_per_mri'] // config.hyperparameters['batch_size'];
        curr_loss = 0;
        for s in range(steps):
            curr_mri = mri[s*config.hyperparameters['batch_size']:(s+1)*config.hyperparameters['batch_size']]
            curr_mri_noisy = mri_noisy[s*config.hyperparameters['batch_size']:(s+1)*config.hyperparameters['batch_size']]
            curr_heatmap = heatmap[s*config.hyperparameters['batch_size']:(s+1)*config.hyperparameters['batch_size']]

            assert not torch.any(torch.isnan(curr_mri)) or not torch.any(torch.isnan(curr_mri_noisy)) or not torch.any(torch.isnan(curr_heatmap))
            with torch.cuda.amp.autocast_mode.autocast():

                # hm1 = model(curr_mri, curr_mri_noisy);
                # hm2 = model(curr_mri_noisy, curr_mri);
                # lih1 = F.l1_loss((curr_mri+hm1), curr_mri_noisy);
                # lih2 = F.l1_loss((curr_mri_noisy+hm2), curr_mri);
                # lhh = F.l1_loss((hm1+hm2), torch.zeros_like(hm1));
                # lh1 = F.l1_loss((hm1)*curr_heatmap, torch.zeros_like(hm1));
                # lh2 = F.l1_loss((hm2)*curr_heatmap, torch.zeros_like(hm1));
                # loss = (lih1 + lih2 + lhh + lh1 + lh2)/ config.hyperparameters['virtual_batch_size'];
                hm1 = model(curr_mri, curr_mri_noisy);
                hm2 = model(curr_mri_noisy, curr_mri);
                lhf1 = DiceFocalLoss(sigmoid=True, smooth_dr=1.0, smooth_nr=1.0)(hm1, curr_heatmap);
                lhf2 = DiceFocalLoss(sigmoid=True, smooth_dr=1.0, smooth_nr=1.0)(hm2, curr_heatmap);
               # lhd2 = dice_loss(hm2, curr_heatmap);
                lhh = DiceLoss(batch=True, smooth_dr=1.0, smooth_nr=1.0)(torch.sigmoid(hm1), torch.sigmoid(hm2));
                loss = (lhf1 + lhf2 + lhh)/ config.hyperparameters['virtual_batch_size'];

            scalar.scale(loss).backward();
            curr_loss += loss.item();
            curr_step+=1;
            curr_iou += (1-(DiceLoss(sigmoid=True)(hm1, curr_heatmap)).item());

            if (curr_step) % config.hyperparameters['virtual_batch_size'] == 0:
                scalar.step(optimizer);
                scalar.update();
                
                model.zero_grad(set_to_none = True);
                epoch_loss.append(curr_loss);
                epoch_IoU.append(curr_iou);
                curr_loss = 0;
                curr_step = 0;
                curr_iou = 0;

            pbar.set_description(('%10s' + '%10.4g'*2)%(epoch, np.mean(epoch_loss), np.mean(epoch_IoU)));

    return np.mean(epoch_loss);


def valid_miccai(model, loader):
    print(('\n' + '%10s'*5) %('Epoch', 'Dice', 'Prec', 'Rec', 'F1'));
    pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epoch_dice = [];
    all_gt = [];
    all_pred = [];
    with torch.no_grad():
        for idx, (batch) in pbar:
            mri, mri_noisy, heatmap, gt_lbl, brainmask = batch[0].to('cuda'), batch[1].to('cuda'), batch[2].to('cuda'), batch[3].numpy()[0], batch[4].to('cuda');

            hm1 = model(mri, mri_noisy);
            hm2 = model(mri_noisy, mri);
            pred_lbl_1 = torch.sigmoid(hm1)>0.5;
            pred_lbl_2 = torch.sigmoid(hm2)>0.5;
            pred = pred_lbl_1 * pred_lbl_2 * brainmask;
            pred_lbl = torch.sum(pred).item()>0;
            if gt_lbl == 1:
                dice = DiceLoss()(pred, heatmap);
                epoch_dice.append(dice.item());
            all_gt.append(gt_lbl);
            all_pred.append(pred_lbl);

            prec,rec,f1,_ = precision_recall_fscore_support(all_gt, all_pred, zero_division=0.0,average='binary');


            # #For regression
            # hm2 = model(mri_noisy, mri);
            # lih1 = F.l1_loss((mri+hm1), mri_noisy);
            # lih2 = F.l1_loss((mri_noisy+hm2), mri);
            # lhh = F.l1_loss((hm1+hm2), torch.zeros_like(hm1));
            # lh1 = F.l1_loss((hm1)*heatmap, torch.zeros_like(hm1));
            # lh2 = F.l1_loss((hm2)*heatmap, torch.zeros_like(hm1));
            # total_loss = lih1 + lih2 + lhh + lh1 + lh2;
            # #==============================================

            
            pbar.set_description(('%10s' + '%10.4g'*4)%(epoch, 0 if len(epoch_dice) == 0 else np.mean(epoch_dice), prec, rec, f1));

    return np.mean(epoch_dice);

def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            logger.add_histogram(tag + "/grad", value.grad.cpu(), step)
            logger.add_histogram(tag + "/vals", value.cpu(), step)

if __name__ == "__main__":
    
    #update_folds();
    #update_folds_isbi();
    #cache_mri_gradients();
    #update_folds_miccai();


    RESUME = False;
    model = UNet3D(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2),
        num_res_units=2,
        ).to('cuda')
    
    if RESUME is True:
        ckpt = torch.load('resume.ckpt');
        model.load_state_dict(ckpt['model']);

    model.to('cuda');
    scalar = torch.cuda.amp.grad_scaler.GradScaler();
    optimizer = optim.AdamW(model.parameters(), lr = config.hyperparameters['learning_rate']);

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min= 1e-6);
    summary_writer = SummaryWriter(os.path.join('exp', 'Unet3D-regression'));
    best_loss = 100;
    start_epoch = 0;

    if RESUME is True:
        optimizer.load_state_dict(ckpt['optimizer']);
        lr_scheduler.load_state_dict(ckpt['scheduler']);
        best_loss = ckpt['best_loss'];
        start_epoch = ckpt['epoch'];
        print(f'Resuming from epoch:{start_epoch}');

    train_loader, test_loader = get_loader_miccai(0);
    sample_output_interval = 10;
    for epoch in range(start_epoch, 1000):
        model.train();
        train_loss = train_miccai(model, train_loader, optimizer, scalar); 
        
        model.eval();
        valid_loss = valid_miccai(model, test_loader);
        summary_writer.add_scalar('train/loss', train_loss, epoch);
        summary_writer.add_scalar('valid/loss', valid_loss, epoch);
        if epoch %sample_output_interval == 0:
            print('sampling outputs...');
            save_examples_miccai(model, test_loader);
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch+1
        }
        torch.save(ckpt,'resume.ckpt');
        lr_scheduler.step();

        if best_loss > valid_loss:
            print(f'new best model found: {valid_loss}')
            best_loss = valid_loss;
            torch.save({'model': model.state_dict(), 
                        'best_loss': best_loss,
                        'hp': config.hyperparameters},'best_model.ckpt');





    
