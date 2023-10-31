from data_utils import  get_loader, window_center_adjustment, get_loader_miccai
import cv2
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch
import config
import torch.nn.functional as F
from model_3d import UNet3D, ResUnet3D
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import glob as glob
import nibabel as nib
from skimage.filters import gaussian, sobel, threshold_otsu
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import zoom
import SimpleITK as sitk
from monai.transforms import NormalizeIntensity, RandSpatialCropd
import matplotlib.cm as cm
from skimage.filters import gaussian
from matplotlib.widgets import Slider, Button
from patchify import patchify
import math
import pickle
from monai.losses.dice import DiceLoss, DiceFocalLoss

def valid(model, loader, total_data):
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        counter = 0;
        epoch_dice = [];
        for idx, (batch) in pbar:
            mri, mri_noisy, heatmap, gt_lbl, brainmask = batch[0].to('cuda'), batch[1].to('cuda'), batch[2].to('cuda'), batch[3], batch[4].to('cuda');
           # mri_c, mri_noisy_c, ht_c,_,_ = total_data[counter];

            
            hm1 = model(mri, mri_noisy);
            hm2 = model(mri_noisy, mri);
            pred_lbl_1 = torch.sigmoid(hm1)>0.5;
            pred_lbl_2 = torch.sigmoid(hm2)>0.5;
            pred = (pred_lbl_1 * pred_lbl_2*brainmask);

            
            
            
            hm1 = hm1.detach().cpu().numpy();
            hm2 = hm2.detach().cpu().numpy();
            mri = mri.detach().cpu().numpy();
            mri_noisy = mri_noisy.detach().cpu().numpy();

            dice_c = total_data[counter];
            diff1 = torch.sum(torch.abs(dice_c-pred));
            if diff1!=0:
                print(f"diff1: {diff1}");
            counter +=1;
            if gt_lbl == 1:
                dice = DiceLoss()(pred, heatmap);
                
                epoch_dice.append(dice.item());

                if dice >0.6:
                    heatmap = heatmap.detach().cpu().numpy();
                    pred = pred.detach().cpu().numpy();
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

                        fig.savefig(os.path.join('samples',f'sample_{counter + idx*config.hyperparameters["batch_size"]}_{j}.png'));
                        plt.close("all");
            

        print(np.mean(epoch_dice));

def save_examples(model, batch, name):
    if os.path.exists('samples') is False:
        os.makedirs('samples');
    mri, mri_noisy, mask = batch[0].to('cuda').unsqueeze(dim=1), batch[1].to('cuda').unsqueeze(dim=1), batch[2].to('cuda').unsqueeze(dim=1);
    hm1 = model(mri, mri_noisy);
    hm2 = model(mri_noisy, mri);
    hm1 = hm1 * mask;
    mri_noisy = mri_noisy * mask;
    hm1_np = hm1.permute(0,2,3,1).cpu().detach().numpy().squeeze()
    mri_noisy = mri_noisy.permute(0,2,3,1).cpu().detach().numpy().squeeze()
    mri = mri.permute(0,2,3,1).cpu().detach().numpy().squeeze() 

    fig, ax = plt.subplots(1,3);
    ax[0].imshow(mri, cmap='gray');
    ax[1].imshow(mri_noisy, cmap='gray');
    ax[2].imshow(hm1_np);
    fig.savefig(os.path.join('samples',f'sample_{name}.png'));

    # for i in range(config.BATCH_SIZE):
    #     mri_recon = (mri_noisy+hm2)*0.5+0.5;
    #     mri_noisy_recon = (mri+hm1)*0.5+0.5;
    #     mri = (mri)*0.5+0.5;
    #     mri_noisy = (mri_noisy)*0.5+0.5;
    #     grid = make_grid([hm1[i], hm2[i], (mri)[i], (mri_noisy)[i], (mri_noisy_recon)[i], (mri_recon)[i]],nrow=2);
    #     save_image(grid, os.path.join('samples',f'sample_{name}_{i}.png'))

def load_and_resample_nii_image(nii_image, image_data, target_spacing=None):
    original_spacing = nii_image.header.get_zooms()[:3]
    target_spacing = np.asarray(target_spacing, dtype=np.float32)
    resample_factor = original_spacing / target_spacing
    new_shape = (np.array(image_data.shape) * resample_factor).astype(int)
    resampled_image_data = zoom(image_data, resample_factor, order=1)
    return resampled_image_data

def rigid_registration(fixed_image, moving_image):
    registration_method = sitk.ImageRegistrationMethod()

    # Set the similarity metric
    registration_method.SetMetricAsMeanSquares()

    # Set the optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=100)

    # Set the initial transformation to identity
    initial_transform = sitk.TranslationTransform(fixed_image.GetDimension())
    registration_method.SetInitialTransform(initial_transform)

    # Perform the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Transform the moving image
    registered_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    return registered_image, final_transform

def histogram_match(source_image, target_image):
    matched_image = sitk.HistogramMatching(source_image, target_image, numberOfHistogramLevels=1024, numberOfMatchPoints=7)
    return matched_image

def predict_on_mri_2d(first_mri_path, second_mri_path, model):
    fixed_path = first_mri_path
    moving_path = second_mri_path


    fixed_image_nib = nib.load(fixed_path)
    moving_image_nib = nib.load(moving_path)

    fixed_image_data = fixed_image_nib.get_fdata()
    moving_image_data = moving_image_nib.get_fdata()

    target_spacing = (1.0, 1.0, 1.0)

    fixed_image_data = load_and_resample_nii_image(fixed_image_nib, fixed_image_data, target_spacing)
    moving_image_data = load_and_resample_nii_image(moving_image_nib, moving_image_data, target_spacing)

    fixed_sitk = sitk.GetImageFromArray(fixed_image_data)
    moving_sitk = sitk.GetImageFromArray(moving_image_data)

    # We perform the histogram matching
    matched_moving_sitk = histogram_match(moving_sitk, fixed_sitk)

    # We convert the matched image back to numpy array
    matched_moving_image_data = sitk.GetArrayFromImage(matched_moving_sitk)

    rigid_registered_image, rigid_transform = rigid_registration(fixed_sitk, matched_moving_sitk)
    rigid_registered_image_data = sitk.GetArrayFromImage(rigid_registered_image)

    fixed_image_data = window_center_adjustment(fixed_image_data);
    rigid_registered_image_data = window_center_adjustment(rigid_registered_image_data);
    r = np.random.randint(fixed_image_data.shape[0]);
    one = fixed_image_data[r,:,:];
    one = cv2.resize(one, (512,512));
    
    two = rigid_registered_image_data[r,:,:];
    two = cv2.resize(two, (512,512));

    diff = two - one;

    transforms = A.Compose([A.Normalize(0.5, 0.5), ToTensorV2()]);
    one = transforms(image = one)['image'];
    two = transforms(image = two)['image'];

    mri, mri_noisy = one.to('cuda').unsqueeze(dim=1), two.to('cuda').unsqueeze(dim=1);
    hm1 = model(mri, mri_noisy);
    hm2 = model(mri_noisy, mri);
    hm1 = hm1;
    mri_noisy = mri_noisy;
    hm1_np = hm2.permute(0,2,3,1).cpu().detach().numpy().squeeze()
    mri_noisy = mri_noisy.permute(0,2,3,1).cpu().detach().numpy().squeeze()
    mri = mri.permute(0,2,3,1).cpu().detach().numpy().squeeze()
    mri = mri * 0.5 + 0.5;
    #mri_noisy = mri * 0.5 + 0.5;
    mri_noisy_rec = mri_noisy + hm1_np;
    #hm1_np = hm1_np > 0;
    fig, ax = plt.subplots(1,5);
    ax[0].imshow(mri, cmap='gray');
    ax[1].imshow(mri_noisy, cmap='gray');
    ax[4].imshow(mri_noisy_rec, cmap='gray');
    ax[2].imshow(hm1_np, cmap='hot');
    ax[3].imshow(diff);
    plt.show();
    
def normalize(img):
    img = (img - np.min(img));
    img = img/np.max(img);
    return img;


def predict_on_mri_3d(first_mri_path, second_mri_path, model, use_cached = False):
    file_name = os.path.basename(first_mri_path);
    file_name = file_name[:file_name.find('.')];
    
    if use_cached is False:
        crop = RandSpatialCropd(keys=['image1','image2', 'mask1', 'mask2'],roi_size= (config.hyperparameters['crop_size_w'],
                                                                                      config.hyperparameters['crop_size_h'],
                                                                                      config.hyperparameters['crop_size_d']),random_size=False);
        normalize_internsity = NormalizeIntensity(subtrahend=0.5, divisor=0.5);
        def preprocess(mrimage):
            mask = mrimage > threshold_otsu(mrimage);
            mask = np.expand_dims(mask, axis=0);
            mrimage = np.expand_dims(mrimage, axis=0);
            mrimage = mrimage / np.max(mrimage);
            mrimage = normalize_internsity(mrimage)[0];
            return mrimage.unsqueeze(dim=0), mask; 

        with torch.no_grad():
            fixed_path = first_mri_path
            moving_path = second_mri_path


            fixed_image_nib = nib.load(fixed_path)
            moving_image_nib = nib.load(moving_path)

            moving_image_nib = nib.as_closest_canonical(moving_image_nib);
            fixed_image_nib = nib.as_closest_canonical(fixed_image_nib);

            fixed_image_data = fixed_image_nib.get_fdata()
            moving_image_data = moving_image_nib.get_fdata()

            target_spacing = (1.0, 1.0, 1.0)

            fixed_image_data = load_and_resample_nii_image(fixed_image_nib, fixed_image_data, target_spacing)
            moving_image_data = load_and_resample_nii_image(moving_image_nib, moving_image_data, target_spacing)

            fixed_sitk = sitk.GetImageFromArray(fixed_image_data)
            moving_sitk = sitk.GetImageFromArray(moving_image_data)

            # We perform the histogram matching
            matched_moving_sitk = histogram_match(moving_sitk, fixed_sitk)

            # We convert the matched image back to numpy array
            matched_moving_image_data = sitk.GetArrayFromImage(matched_moving_sitk)

            rigid_registered_image, rigid_transform = rigid_registration(fixed_sitk, matched_moving_sitk)
            rigid_registered_image_data = sitk.GetArrayFromImage(rigid_registered_image)

            fixed_image_data = window_center_adjustment(fixed_image_data);
            rigid_registered_image_data = window_center_adjustment(rigid_registered_image_data);

            fixed_image_data, fixed_image_data_mask = preprocess(fixed_image_data);
            rigid_registered_image_data, rigid_registered_image_data_mask = preprocess(rigid_registered_image_data);

            fixed_image_data = fixed_image_data.squeeze();
            rigid_registered_image_data = rigid_registered_image_data.squeeze();

            w,h,d = fixed_image_data.shape;
            new_w = math.ceil(w / config.hyperparameters['crop_size_w']) * config.hyperparameters['crop_size_w'];
            new_h = math.ceil(h / config.hyperparameters['crop_size_h']) * config.hyperparameters['crop_size_h'];
            new_d = math.ceil(d / config.hyperparameters['crop_size_d']) * config.hyperparameters['crop_size_d'];

            fixed_image_data_padded  = torch.zeros((new_w, new_h, new_d), dtype = fixed_image_data.dtype);
            rigid_registered_image_data_padded  = torch.zeros((new_w, new_h, new_d), dtype = fixed_image_data.dtype);

            fixed_image_data_padded[:w,:h,:d] = fixed_image_data;
            rigid_registered_image_data_padded[:w,:h,:d] = rigid_registered_image_data;

            step_w, step_h, step_d = config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d'];
            fixed_image_data_patches = patchify(fixed_image_data_padded.numpy(), 
                                                (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                (step_w, step_h, step_d));
            
            rigid_registered_image_data_patches = patchify(rigid_registered_image_data_padded.numpy(), 
                                                (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                (step_w, step_h, step_d));

            predicted_mri = np.zeros((new_w, new_h, new_d,3), dtype = np.float64);
            predicted_mri_noisy = np.zeros((new_w, new_h, new_d), dtype = np.float64);
            predicted_positive_thresh = np.zeros((new_w, new_h, new_d,1), dtype = np.float64);
            predicted_negative_thresh = np.zeros((new_w, new_h, new_d,1), dtype = np.float64);
            predicted_hm1_color = np.zeros((new_w, new_h, new_d,4), dtype = np.float64);
            for i in range(fixed_image_data_patches.shape[0]):
                for j in range(fixed_image_data_patches.shape[1]):
                    for k in range(fixed_image_data_patches.shape[2]):
                        #trans_ret = crop({'image1': fixed_image_data, 'image2': rigid_registered_image_data, 'mask1': fixed_image_data_mask, 'mask2': rigid_registered_image_data_mask});

                        fixed_image_data_trans = torch.from_numpy(fixed_image_data_patches[i,j,k,:,:,:]);
                        rigid_registered_image_data_trans = torch.from_numpy(rigid_registered_image_data_patches[i,j,k,:,:,:]);

                        

                        mri, mri_noisy = fixed_image_data_trans.to('cuda').unsqueeze(dim=0).unsqueeze(dim=0), rigid_registered_image_data_trans.to('cuda').unsqueeze(dim=0).unsqueeze(dim=0);
                        #mri_mask, mri_noisy_mask = fixed_image_data_mask_trans.to('cuda'), rigid_registered_image_data_mask_trans.to('cuda');
                        hm1 = model(mri, mri_noisy);
                        hm2 = model(mri_noisy, mri);
                        # hm1 = hm1 *2.0;
                        # hm2 = hm2 *2.0;

                        mri_recon = (mri_noisy+hm2);
                        mri_noisy_recon = (mri+hm1);

                        hm1 = hm1.cpu().detach().numpy().squeeze()
                        hm2 = hm2.cpu().detach().numpy().squeeze()
                        mri_noisy = mri_noisy.cpu().detach().numpy().squeeze()
                        mri = mri.cpu().detach().numpy().squeeze()
                        mri_noisy_recon = mri_noisy_recon.cpu().detach().numpy().squeeze()
                        mri_recon = mri_recon.cpu().detach().numpy().squeeze()
                        hm1_diff = mri_noisy - mri;

                        hm1_normalize = normalize(hm1);
                        hm1_color = cm.coolwarm(normalize(hm1));

                        #mri = np.repeat(np.expand_dims(mri, axis=-1), axis=-1, repeats=3);
                        mri = normalize(mri);

                        hm1_positive_thresh = (hm1 > 0.1);
                        hm1_negative_thresh = (hm1 < -0.1);
                        hm_thresh = np.clip(hm1_negative_thresh+hm1_positive_thresh, 0, 1);
                        hm1_color_positive = hm1_color * np.expand_dims(hm1_positive_thresh,axis=-1);
                        hm1_color_positive = hm1_color_positive[:,:,:,:3];
                        hm1_color_negative = hm1_color * np.expand_dims(hm1_negative_thresh,axis=-1);
                        hm1_color_negative = hm1_color_negative[:,:,:,:3];
                        
                        
                        hm1_positive = (hm1_positive_thresh * hm1);
                        hm1_negative = ((hm1_negative_thresh*-1)*hm1);

                        hm1_positive = np.expand_dims(hm1_positive, axis = -1);
                        hm1_negative = np.expand_dims(hm1_negative, axis = -1);

                        mri = np.repeat(np.expand_dims(mri, axis=-1), axis=-1, repeats=3);

                        hm1_positive_thresh = gaussian(hm1_positive_thresh,1);
                        hm1_negative_thresh = gaussian(hm1_negative_thresh,1);

                        hm1_positive_thresh = np.expand_dims(hm1_positive_thresh, axis=-1)
                        hm1_negative_thresh = np.expand_dims(hm1_negative_thresh, axis=-1)

                        
                        
                        #add mri to the whole predicted mri
                        predicted_mri[i*step_w:(i)*step_w + config.hyperparameters['crop_size_w'], 
                                    j*step_h:(j)*step_h + config.hyperparameters['crop_size_h'], 
                                    k*step_d:(k)*step_d + config.hyperparameters['crop_size_d'],:] = mri;
                        
                        predicted_mri_noisy[i*step_w:(i)*step_w + + config.hyperparameters['crop_size_w'], 
                                    j*step_h:(j)*step_h+ config.hyperparameters['crop_size_h'], 
                                    k*step_d:(k)*step_d+ config.hyperparameters['crop_size_d']] = mri_noisy;
                        
                        predicted_negative_thresh[i*step_w:(i)*step_w + + config.hyperparameters['crop_size_w'], 
                                    j*step_h:(j)*step_h+ config.hyperparameters['crop_size_h'], 
                                    k*step_d:(k)*step_d+ config.hyperparameters['crop_size_d'],:] = hm1_negative_thresh;
                        
                        predicted_positive_thresh[i*step_w:(i)*step_w + + config.hyperparameters['crop_size_w'], 
                                    j*step_h:(j)*step_h+ config.hyperparameters['crop_size_h'], 
                                    k*step_d:(k)*step_d+ config.hyperparameters['crop_size_d'],:] = hm1_positive_thresh;
                        
                        predicted_hm1_color[i*step_w:(i)*step_w + + config.hyperparameters['crop_size_w'], 
                                    j*step_h:(j)*step_h+ config.hyperparameters['crop_size_h'], 
                                    k*step_d:(k)*step_d+ config.hyperparameters['crop_size_d'],:] = hm1_color;

            predicted_mri = predicted_mri[:w,:h,:d,:];
            predicted_mri_noisy = predicted_mri_noisy[:w,:h,:d];
            predicted_negative_thresh = predicted_negative_thresh[:w,:h,:d,:];
            predicted_positive_thresh = predicted_positive_thresh[:w,:h,:d,:];
            predicted_hm1_color = predicted_hm1_color[:w,:h,:d,:];

            pickle.dump([predicted_mri, 
                         predicted_mri_noisy, 
                         predicted_negative_thresh, 
                         predicted_positive_thresh, 
                         predicted_hm1_color], 
                         open(f'{file_name}.dmp', 'wb'));

    else:
        predicted_mri, predicted_mri_noisy, predicted_negative_thresh, predicted_positive_thresh, predicted_hm1_color = pickle.load(open(f'{file_name}.dmp', 'rb'))    

    fig,ax = plt.subplots(2,3);
    # subfigures = fig.subfigures(2,1);
    # axes1 = subfigures[0].subplots(1,3);
    # subfigures[0].suptitle('Old MRI');
    # axes2 = subfigures[1].subplots(1,3);
    # subfigures[1].suptitle('New MRI');

    #This is heatmap drawing
    # axes1[0].imshow(hm1[pos_cords[0][r], :, :], cmap = 'hot');
    # axes1[1].imshow(hm1[:,pos_cords[1][r], :], cmap = 'hot');
    # axes1[2].imshow(hm1[ :, :,pos_cords[2][r]], cmap = 'hot');

    # axes1[0].axis('off');
    # axes1[1].axis('off');
    # axes1[2].axis('off');


    # axes2[0].imshow(hm1_color[pos_cords[0][r], :, :]);
    # axes2[1].imshow(hm1_color[:,pos_cords[1][r], :]);
    # axes2[2].imshow(hm1_color[ :, :,pos_cords[2][r]]);
    # axes2[0].axis('off');
    # axes2[1].axis('off');
    # axes2[2].axis('off');
    #============================

    global x,y,z;
    x = 10;
    y = 10;
    z = 10;
    intensity_scale = 1.0;

    global mri_highlighted;
    mri_highlighted = (1-predicted_negative_thresh)*predicted_mri + (predicted_negative_thresh * (predicted_hm1_color[:,:,:,:3]*intensity_scale + predicted_mri*(1-intensity_scale)));
    mri_highlighted = (1-predicted_positive_thresh)*mri_highlighted + (predicted_positive_thresh * (predicted_hm1_color[:,:,:,:3]*intensity_scale + mri_highlighted*(1-intensity_scale)));
    ax[0][0].imshow(mri_highlighted[x, :, :], cmap='hot');
    ax[0][1].imshow(mri_highlighted[:,y, :], cmap='hot');
    ax[0][2].imshow(mri_highlighted[ :, :,z], cmap='hot');
    ax[0][0].axis('off');
    ax[0][1].axis('off');
    ax[0][2].axis('off');


    ax[1][0].imshow(predicted_mri_noisy[x, :, :], cmap='gray');
    ax[1][1].imshow(predicted_mri_noisy[:,y, :], cmap='gray');
    ax[1][2].imshow(predicted_mri_noisy[ :, :,z], cmap='gray');
    ax[1][0].axis('off');
    ax[1][1].axis('off');
    ax[1][2].axis('off');

    fig.subplots_adjust(left=0.25, bottom=0.25)

    axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    amp_slider = Slider(
        ax=axamp,
        label="Alpha",
        valmin=0.0,
        valmax=1,
        valinit=intensity_scale,
        orientation="vertical"
    )

    ax_x = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    x_slider = Slider(
        ax=ax_x,
        label='X',
        valmin=0,
        valmax=predicted_mri.shape[0]-1,
        valinit=x,
    )

    ax_y = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    y_slider = Slider(
        ax=ax_y,
        label='Y',
        valmin=0,
        valmax=predicted_mri.shape[1]-1,
        valinit=y,
    )

    ax_z = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    z_slider = Slider(
        ax=ax_z,
        label='Z',
        valmin=0,
        valmax=predicted_mri.shape[2]-1,
        valinit=z,
    )

    def update_alpha(val):
        global mri_highlighted;
        mri_highlighted = (1-predicted_negative_thresh)*predicted_mri + (predicted_negative_thresh * (predicted_hm1_color[:,:,:,:3]*val + predicted_mri*(1-val)));
        mri_highlighted = (1-predicted_positive_thresh)*mri_highlighted + (predicted_positive_thresh * (predicted_hm1_color[:,:,:,:3]*val + mri_highlighted*(1-val)));
        ax[0][0].imshow(mri_highlighted[x, :, :]);
        ax[0][1].imshow(mri_highlighted[:,y, :]);
        ax[0][2].imshow(mri_highlighted[ :, :,z]);

    def update(val):
        update_alpha(val);
    
    def update_slice_x(val):
        global x;
        x = int(val);
        ax[0][0].imshow(mri_highlighted[x, :, :]);
        ax[0][1].imshow(mri_highlighted[:,y, :]);
        ax[0][2].imshow(mri_highlighted[ :, :,z]);

        ax[1][0].imshow(predicted_mri_noisy[x, :, :], cmap='gray');
        ax[1][1].imshow(predicted_mri_noisy[:,y, :], cmap='gray');
        ax[1][2].imshow(predicted_mri_noisy[ :, :,z], cmap='gray');
  

    def update_slice_y(val):
        global y;
        y = int(val);
        ax[0][0].imshow(mri_highlighted[x, :, :]);
        ax[0][1].imshow(mri_highlighted[:,y, :]);
        ax[0][2].imshow(mri_highlighted[ :, :,z]);

        ax[1][0].imshow(predicted_mri_noisy[x, :, :], cmap='gray');
        ax[1][1].imshow(predicted_mri_noisy[:,y, :], cmap='gray');
        ax[1][2].imshow(predicted_mri_noisy[ :, :,z], cmap='gray');

    
    def update_slice_z(val):
        global z;
        z = int(val);
        ax[0][0].imshow(mri_highlighted[x, :, :]);
        ax[0][1].imshow(mri_highlighted[:,y, :]);
        ax[0][2].imshow(mri_highlighted[ :, :,z]);

        ax[1][0].imshow(predicted_mri_noisy[x, :, :], cmap='gray');
        ax[1][1].imshow(predicted_mri_noisy[:,y, :], cmap='gray');
        ax[1][2].imshow(predicted_mri_noisy[ :, :,z], cmap='gray');


    amp_slider.on_changed(update)
    x_slider.on_changed(update_slice_x)
    y_slider.on_changed(update_slice_y)
    z_slider.on_changed(update_slice_z)
    plt.show();

def predict_on_lesjak(base_path, first_mri_path, second_mri_path, model, use_cached = False):
    file_name = os.path.basename(first_mri_path);
    file_name = file_name[:file_name.find('.')];

    total_dice = [];
    if use_cached is False:
        crop = RandSpatialCropd(keys=['image1','image2', 'mask1', 'mask2'],roi_size= (config.hyperparameters['crop_size_w'],
                                                                                      config.hyperparameters['crop_size_h'],
                                                                                      config.hyperparameters['crop_size_d']),random_size=False);
        normalize_internsity = NormalizeIntensity(subtrahend=0.5, divisor=0.5);
        def preprocess(mrimage):
            mask = mrimage > threshold_otsu(mrimage);
            mask = np.expand_dims(mask, axis=0);
            mrimage = mrimage / (np.max(mrimage)+1e-4);
            return mrimage, mask; 

        counter = 0;
        with torch.no_grad():
            gt_path = base_path + "\\ground_truth.nii.gz";
            brainmask_path = base_path + "\\brain_mask.nii.gz";


            mri1 = nib.load(first_mri_path)
            mri2 = nib.load(second_mri_path)
            gt_image_nib = nib.load(gt_path)
            brainmask_image_nib = nib.load(brainmask_path)


            mri1 = mri1.get_fdata()
            mri2 = mri2.get_fdata()
            gt = gt_image_nib.get_fdata();
            brainmask = brainmask_image_nib.get_fdata();

            gt = gt*brainmask;


            mri1 = window_center_adjustment(mri1);
            mri2 = window_center_adjustment(mri2);

            mri1, fixed_image_data_mask = preprocess(mri1);
            mri2, rigid_registered_image_data_mask = preprocess(mri2);


            w,h,d = mri1.shape;
            new_w = math.ceil(w / config.hyperparameters['crop_size_w']) * config.hyperparameters['crop_size_w'];
            new_h = math.ceil(h / config.hyperparameters['crop_size_h']) * config.hyperparameters['crop_size_h'];
            new_d = math.ceil(d / config.hyperparameters['crop_size_d']) * config.hyperparameters['crop_size_d'];

            mri1_padded  = np.zeros((new_w, new_h, new_d), dtype = mri1.dtype);
            mri2_padded  = np.zeros((new_w, new_h, new_d), dtype = mri2.dtype);
            gt_padded  = np.zeros((new_w, new_h, new_d), dtype = gt.dtype);
            brainmask_padded  = np.zeros((new_w, new_h, new_d), dtype = brainmask.dtype);

            mri1_padded[:w,:h,:d] = mri1;
            mri2_padded[:w,:h,:d] = mri2;
            gt_padded[:w,:h,:d] = gt;
            brainmask_padded[:w,:h,:d] = brainmask;

            step_w, step_h, step_d = config.hyperparameters['crop_size_w']//2, config.hyperparameters['crop_size_h']//2, config.hyperparameters['crop_size_d']//2;
            mri1_patches = patchify(mri1_padded, 
                                                (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                (step_w, step_h, step_d));
            mri2_patches = patchify(mri2_padded, 
                                                (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                (step_w, step_h, step_d));
            gt_patches = patchify(gt_padded, 
                                                (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                (step_w, step_h, step_d));
            brainmask_patches = patchify(brainmask_padded, 
                                                (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                (step_w, step_h, step_d));

            # mri1_patches = mri1_patches.reshape(mri1_patches.shape[0]*mri1_patches.shape[1]*mri1_patches.shape[2], mri1_patches.shape[3], mri1_patches.shape[4],mri1_patches.shape[5]);
            # mri2_patches = mri2_patches.reshape(mri2_patches.shape[0]*mri2_patches.shape[1]*mri2_patches.shape[2], mri2_patches.shape[3], mri2_patches.shape[4],mri2_patches.shape[5]);
            # gt_patches = gt_patches.reshape(gt_patches.shape[0]*gt_patches.shape[1]*gt_patches.shape[2], gt_patches.shape[3], gt_patches.shape[4],gt_patches.shape[5]);
            # brainmask_patches = brainmask_patches.reshape(brainmask_patches.shape[0]*brainmask_patches.shape[1]*brainmask_patches.shape[2], brainmask_patches.shape[3], brainmask_patches.shape[4],brainmask_patches.shape[5]);
            predicted_aggregated = np.zeros((new_w, new_h, new_d), dtype = np.int32);
            predicted_aggregated_count = np.zeros((new_w, new_h, new_d), dtype = np.int32);
            # # data = [];
            # for i in range(mri2_patches.shape[0]):
            #     if np.sum(mri1_patches[i]) != 0 and np.sum(mri2_patches[i]) != 0:
            #         data.append((mri1_patches[i],mri2_patches[i],gt_patches[i], 0 if np.sum(gt_patches[i]) == 0 else 1, brainmask_patches[i]))

            # return data;
            total_pred = [];
            for i in range(mri1_patches.shape[0]):
                for j in range(mri1_patches.shape[1]):
                    for k in range(mri1_patches.shape[2]):
                #         #trans_ret = crop({'image1': fixed_image_data, 'image2': rigid_registered_image_data, 'mask1': fixed_image_data_mask, 'mask2': rigid_registered_image_data_mask});
                #         if np.sum(mri1_patches[i,j,k,:,:,:]) != 0 and np.sum(mri2_patches[i,j,k,:,:,:]) != 0:
                            
                        
                # mri1_patches_trans = torch.from_numpy(mri1_patches[i,j,k,:,:,:]);
                # mri2_patches_trans = torch.from_numpy(mri2_patches[i,j,k,:,:,:]);
                # gt_lbl = 0 if np.sum(gt_patches[i,j,k,:,:,:]) == 0 else 1;
                # heatmap = torch.from_numpy(gt_patches[i,j,k,:,:,:]);
                # brainmask = torch.from_numpy(brainmask_patches[i,j,k,:,:,:]).to('cuda');

                # mri1_patches_trans = normalize_internsity(mri1_patches_trans);

                # #mri2_c = self.augment_noisy_image(mri2_c);
                # mri2_patches_trans = normalize_internsity(mri2_patches_trans);

                        mri1, mri2, ret_gt, gt_lbl, brainmask = mri1_patches[i,j,k,:,:,:], mri2_patches[i,j,k,:,:,:], gt_patches[i,j,k,:,:,:], 0 if np.sum(gt_patches[i,j,k,:,:,:]) == 0 else 1, brainmask_patches[i,j,k,:,:,:];

                        mri1 = np.expand_dims(mri1, axis=0);
                        mri2 = np.expand_dims(mri2, axis=0);
                        ret_gt = np.expand_dims(ret_gt, axis=0);

                        ret_mri1 = normalize_internsity(mri1);
                        ret_mri2 = normalize_internsity(mri2);

                        ret_gt = torch.from_numpy(ret_gt).to('cuda').unsqueeze(dim=0)
                        brainmask = torch.from_numpy(brainmask).to('cuda');

                        

                        mri, mri_noisy = ret_mri1.to('cuda').unsqueeze(dim=0), ret_mri2.to('cuda').unsqueeze(dim=0);
                        #mri_mask, mri_noisy_mask = fixed_image_data_mask_trans.to('cuda'), rigid_registered_image_data_mask_trans.to('cuda');
                        hm1 = model(mri, mri_noisy);
                        hm2 = model(mri_noisy, mri);
                        pred_lbl_1 = torch.sigmoid(hm1)>0.5;
                        pred_lbl_2 = torch.sigmoid(hm2)>0.5;
                        pred = pred_lbl_1 * pred_lbl_2 * brainmask;

                        s = torch.sum(pred);

                        #gt_lbl = torch.sum(pred).item()>0;
                        if gt_lbl == 1:
                            dice = DiceLoss()(pred, ret_gt);
                            total_dice.append(dice.item());
                        if gt_lbl ==0 and s > 0:
                            print('a');
                        
                        hm1 = hm1.detach().cpu().numpy();
                        hm2 = hm2.detach().cpu().numpy();
                        mri = mri.detach().cpu().numpy();
                        mri_noisy = mri_noisy.detach().cpu().numpy();

                        predicted_aggregated[i*step_w:i*step_w + config.hyperparameters['crop_size_w'], 
                                    j*step_h:(j)*step_h + config.hyperparameters['crop_size_h'], 
                                    k*step_d:(k)*step_d + config.hyperparameters['crop_size_d']] += np.array(pred.squeeze().detach().cpu().numpy()).astype("int32");
                
                        predicted_aggregated_count[i*step_w:i*step_w + config.hyperparameters['crop_size_w'], 
                                    j*step_h:(j)*step_h + config.hyperparameters['crop_size_h'], 
                                    k*step_d:(k)*step_d + config.hyperparameters['crop_size_d']] += np.ones((config.hyperparameters['crop_size_w'], 
                                                                                                            config.hyperparameters['crop_size_h'], 
                                                                                                            config.hyperparameters['crop_size_d']), dtype=np.int32);
                
                # if gt_lbl > 0:
                #     heatmap = heatmap.detach().cpu().numpy();
                #     pred = pred.detach().cpu().numpy();
                #     for j in range(1):
                #         #heatmap = (1-heatmap) > 0;
                #         pos_cords = np.where(heatmap >0);
                #         if len(pos_cords[0]) != 0:
                #             r = np.random.randint(0,len(pos_cords[0]));
                #             center = [pos_cords[0][r], pos_cords[1][r],pos_cords[2][r]]
                #         else:
                #             center = [hm1.shape[2]//2, hm1.shape[3]//2, hm1.shape[4]//2]
                #         fig, ax = plt.subplots(2,6);
                #         ax[0][0].imshow(pred[0,0,center[0], :, :], cmap='hot');
                #         ax[0][1].imshow(pred[0,0,:,center[1], :], cmap='hot');
                #         ax[0][2].imshow(pred[0,0, :, :,center[2]], cmap='hot');

                #         ax[0][3].imshow(heatmap[center[0], :, :]);
                #         ax[0][4].imshow(heatmap[:,center[1], :]);
                #         ax[0][5].imshow(heatmap[:, :,center[2]]);

                #         ax[1][0].imshow(mri[0,0,center[0], :, :], cmap='gray');
                #         ax[1][1].imshow(mri[0,0,:,center[1], :], cmap='gray');
                #         ax[1][2].imshow(mri[0,0, :, :,center[2]], cmap='gray');

                #         ax[1][3].imshow(mri_noisy[0,0,center[0], :, :], cmap='gray');
                #         ax[1][4].imshow(mri_noisy[0,0,:,center[1], :], cmap='gray');
                #         ax[1][5].imshow(mri_noisy[0,0, :, :,center[2]], cmap='gray');

                #         fig.savefig(os.path.join('samples',f'sample_{counter}_{j}.png'));
                #         plt.close("all");

        
        #m = np.max(predicted_aggregated_count);                
        final_pred = np.where(predicted_aggregated>predicted_aggregated_count//2,1,0);
        gt_padded = torch.from_numpy(gt_padded)
        final_pred = torch.from_numpy(final_pred)
        gt_lbl = torch.sum(gt_padded);
        dice = DiceLoss()(final_pred.unsqueeze(0).unsqueeze(0), gt_padded.unsqueeze(0).unsqueeze(0));
        d = np.mean(total_dice);
        return dice, gt_lbl;

if __name__ == "__main__":
    #cache_dataset();
    # reader = sitk.ImageSeriesReader()

    # dicom_names = reader.GetGDCMSeriesFileNames('C:\\PhD\\Thesis\\MRI Project\\BRATS\\00000\\T1wCE');
    # reader.SetFileNames(dicom_names)

    # image = reader.Execute()
    # sitk.WriteImage(image, f'test.nii.gz')

    model = UNet3D(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=2,
            );
    total_parameters = sum(p.numel() for p in model.parameters());
    ckpt = torch.load('best_model-miccai.ckpt');
    model.load_state_dict(ckpt['model']);
    model.to('cuda');

    train_ids, test_ids = pickle.load(open(os.path.join(f'cache_miccai',f'{0}.fold'), 'rb'));

    model.eval();
    #predict_on_mri_3d('mri_data\\TUM20-20170928.nii.gz', 'mri_data\\TUM20-20180402.nii.gz', model, use_cached=False);
    total_dice = [];
    for i in range(5,len(test_ids)):
        dice, gt_lbl = predict_on_lesjak(test_ids[i], f'{test_ids[i]}\\flair_time01_on_middle_space.nii.gz',
                       f'{test_ids[i]}\\flair_time02_on_middle_space.nii.gz', model, use_cached=False);
        #total_data.extend(data);
        if gt_lbl.item() > 0:
            if dice < 0.98:
                total_dice.append(dice);
    print(np.mean(total_dice));

    train_loader, test_loader = get_loader_miccai(0);
    
    # model.eval();
    valid_loss = valid(model, test_loader, total_data);
