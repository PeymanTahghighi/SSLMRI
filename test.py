from data_utils import  get_loader, window_center_adjustment
import cv2
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch
import config
import torch.nn.functional as F
from model_3d import UNet3D,  AttenUnet3D, ResUnet3D
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

def valid(model, loader):
    print(('\n' + '%10s'*1) %('Loss'));
    pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epoch_loss = [];
   
    for idx, (batch) in pbar:
        mri, mri_noisy, mask = batch[0].to('cuda').unsqueeze(dim=1), batch[1].to('cuda').unsqueeze(dim=1), batch[2].to('cuda').unsqueeze(dim=1);
        hm1 = model(mri, mri_noisy);
        hm2 = model(mri_noisy, mri);
        lih1 = F.l1_loss((mri+hm1)*mask.float(), mri_noisy*mask.float());
        lih2 = F.l1_loss((mri_noisy+hm2)*mask.float(), mri*mask.float());
        lhh = F.l1_loss((hm1+hm2)*mask.float(), torch.zeros_like(hm1));
        total_loss = lih1 + lih2 + lhh;
        epoch_loss.append(total_loss.item());
        save_examples(model, batch, idx);
        pbar.set_description(('%10.4g')%(np.mean(epoch_loss)));

    return np.mean(epoch_loss);

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
        crop = RandSpatialCropd(keys=['image1','image2', 'mask1', 'mask2'],roi_size= (config.CROP_SIZE_W,config.CROP_SIZE_H,config.CROP_SIZE_D),random_size=False);
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
            new_w = math.ceil(w / config.CROP_SIZE_W) * config.CROP_SIZE_W;
            new_h = math.ceil(h / config.CROP_SIZE_H) * config.CROP_SIZE_H;
            new_d = math.ceil(d / config.CROP_SIZE_D) * config.CROP_SIZE_D;

            fixed_image_data_padded  = torch.zeros((new_w, new_h, new_d), dtype = fixed_image_data.dtype);
            rigid_registered_image_data_padded  = torch.zeros((new_w, new_h, new_d), dtype = fixed_image_data.dtype);

            fixed_image_data_padded[:w,:h,:d] = fixed_image_data;
            rigid_registered_image_data_padded[:w,:h,:d] = rigid_registered_image_data;


            fixed_image_data_patches = patchify(fixed_image_data_padded.numpy(), 
                                                (config.CROP_SIZE_W, config.CROP_SIZE_H, config.CROP_SIZE_D), 
                                                (config.CROP_SIZE_W, config.CROP_SIZE_H, config.CROP_SIZE_D));
            
            rigid_registered_image_data_patches = patchify(rigid_registered_image_data_padded.numpy(), 
                                                (config.CROP_SIZE_W, config.CROP_SIZE_H, config.CROP_SIZE_D), 
                                                (config.CROP_SIZE_W, config.CROP_SIZE_H, config.CROP_SIZE_D));

            predicted_mri = np.zeros((new_w, new_h, new_d,3), dtype = np.float64);
            predicted_mri_noisy = np.zeros((new_w, new_h, new_d), dtype = np.float64);
            predicted_positive_thresh = np.zeros((new_w, new_h, new_d,1), dtype = np.float64);
            predicted_negative_thresh = np.zeros((new_w, new_h, new_d,1), dtype = np.float64);
            predicted_hm1_color = np.zeros((new_w, new_h, new_d,4), dtype = np.float64);
            for i in range(math.ceil(w / config.CROP_SIZE_W)):
                for j in range(math.ceil(h / config.CROP_SIZE_H)):
                    for k in range(math.ceil(d / config.CROP_SIZE_D)):
                        #trans_ret = crop({'image1': fixed_image_data, 'image2': rigid_registered_image_data, 'mask1': fixed_image_data_mask, 'mask2': rigid_registered_image_data_mask});

                        fixed_image_data_trans = torch.from_numpy(fixed_image_data_patches[i,j,k,:,:,:]);
                        rigid_registered_image_data_trans = torch.from_numpy(rigid_registered_image_data_patches[i,j,k,:,:,:]);

                        

                        mri, mri_noisy = fixed_image_data_trans.to('cuda').unsqueeze(dim=0).unsqueeze(dim=0), rigid_registered_image_data_trans.to('cuda').unsqueeze(dim=0).unsqueeze(dim=0);
                        #mri_mask, mri_noisy_mask = fixed_image_data_mask_trans.to('cuda'), rigid_registered_image_data_mask_trans.to('cuda');
                        hm1 = model(mri, mri_noisy);
                        hm2 = model(mri_noisy, mri);

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
                        predicted_mri[i*config.CROP_SIZE_W:(i+1)*config.CROP_SIZE_W, 
                                    j*config.CROP_SIZE_H:(j+1)*config.CROP_SIZE_H, 
                                    k*config.CROP_SIZE_D:(k+1)*config.CROP_SIZE_D,:] = mri;
                        
                        predicted_mri_noisy[i*config.CROP_SIZE_W:(i+1)*config.CROP_SIZE_W, 
                                    j*config.CROP_SIZE_H:(j+1)*config.CROP_SIZE_H, 
                                    k*config.CROP_SIZE_D:(k+1)*config.CROP_SIZE_D] = mri_noisy;
                        
                        predicted_negative_thresh[i*config.CROP_SIZE_W:(i+1)*config.CROP_SIZE_W, 
                                    j*config.CROP_SIZE_H:(j+1)*config.CROP_SIZE_H, 
                                    k*config.CROP_SIZE_D:(k+1)*config.CROP_SIZE_D,:] = hm1_negative_thresh;
                        
                        predicted_positive_thresh[i*config.CROP_SIZE_W:(i+1)*config.CROP_SIZE_W, 
                                    j*config.CROP_SIZE_H:(j+1)*config.CROP_SIZE_H, 
                                    k*config.CROP_SIZE_D:(k+1)*config.CROP_SIZE_D,:] = hm1_positive_thresh;
                        
                        predicted_hm1_color[i*config.CROP_SIZE_W:(i+1)*config.CROP_SIZE_W, 
                                    j*config.CROP_SIZE_H:(j+1)*config.CROP_SIZE_H, 
                                    k*config.CROP_SIZE_D:(k+1)*config.CROP_SIZE_D,:] = hm1_color;

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
    ax[0][0].imshow(mri_highlighted[x, :, :]);
    ax[0][1].imshow(mri_highlighted[:,y, :]);
    ax[0][2].imshow(mri_highlighted[ :, :,z]);
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

if __name__ == "__main__":
    #cache_dataset();
    # reader = sitk.ImageSeriesReader()

    # dicom_names = reader.GetGDCMSeriesFileNames('C:\\PhD\\Thesis\\MRI Project\\BRATS\\00000\\T1wCE');
    # reader.SetFileNames(dicom_names)

    # image = reader.Execute()
    # sitk.WriteImage(image, f'test.nii.gz')

    model = ResUnet3D()
    total_parameters = sum(p.numel() for p in model.parameters());
    ckpt = torch.load('best_model_1.ckpt');
    model.load_state_dict(ckpt['model']);
    model.to('cuda');

    predict_on_mri_3d('mri_data\\TUM20-20170928.nii.gz', 'mri_data\\TUM20-20180402.nii.gz', model, use_cached=True);

    # train_loader, test_loader = get_loader();
    
    # model.eval();
    # valid_loss = valid(model, test_loader);
