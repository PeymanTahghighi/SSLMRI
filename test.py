from data_utils import window_center_adjustment, get_loader_miccai
import cv2
import numpy as np
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
import glob as glob
import nibabel as nib
from skimage.filters import gaussian, threshold_otsu
import albumentations as A
from albumentations.pytorch import ToTensorV2
import SimpleITK as sitk
from monai.transforms import NormalizeIntensity
import matplotlib.cm as cm
from skimage.filters import gaussian
from matplotlib.widgets import Slider
from patchify import patchify
import math
import pickle
from data_utils import inpaint_3d, add_synthetic_lesion_wm
from VNet import VNet
from copy import copy
import imageio
import argparse

def valid(model, loader, dataset):
    """runs validation on the given loader, calculate F1, HD and Dice measure and return them

    Parameters
    ----------
    model : nn.Module
        model to make predictions from

    loader : DataLoader
        Dataloader of the dataset
    
    dataset: Dataset
        Dataset to utilize for building up final prediction and calculating final results
    """
    print(('\n' + '%10s'*2) %('Epoch', 'Dice'));
    pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    with torch.no_grad():
        for idx, (batch) in pbar:
            mri, mri_noisy, heatmap, brainmask, patient_id, loc = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device), batch[3].to(args.device), batch[4], batch[5];
            volume_batch1 = torch.cat([mri, mri_noisy, mri - mri_noisy], dim=1)
            volume_batch2 = torch.cat([mri_noisy, mri, mri_noisy - mri], dim=1)
            hm1 = model(volume_batch1);
            hm2 = model(volume_batch2);
            pred_lbl_1 = torch.sigmoid(hm1)>0.5;
            pred_lbl_2 = torch.sigmoid(hm2)>0.5;
            pred = pred_lbl_1 * pred_lbl_2 * brainmask;
            dataset.update_prediction(pred, patient_id[0], loc);
    
    res = dataset.calculate_metrics(simple = False);
    return res;
    
def normalize(img):
    """normalize the range of given image between [0,1]

    Parameters
    ----------
    img : np.ndarray
        input image
    """
    img = (img - np.min(img));
    img = img/np.max(img);
    return img;

def save_to_gif(tensors, axis, r):
    """save given predicted tensors to GIF format for illustration purposes

    Parameters
    ----------
    tensors : List
        a list of tensors required to build GIF, includes prediction and highlighted prediction(highlighted changed areas)
    
    axis: Int
        axis to utilize to create GIF, could be 0,1,2 representing x,y,z

    r: List
        specifies the range in which images should be used to create GIF
    """

    def save_slices(name, tensor):
        if os.path.exists(f'slices/{name}0') is False:
            os.mkdir(f'slices/{name}0');
        for i in range(tensor.shape[0]):
            plt.imsave(f'slices/{name}0\\{i}.png', tensor[i,:,:], cmap='gray');
            plt.close('all');
    
        if os.path.exists(f'slices/{name}1') is False:
            os.mkdir(f'slices/{name}1');
        for i in range(tensor.shape[1]):
            plt.imsave(f'slices/{name}1\\{i}.png', tensor[:,i,:], cmap='gray');
            plt.close('all');
        
        if os.path.exists(f'slices/{name}2') is False:
            os.mkdir(f'slices/{name}2');
        for i in range(tensor.shape[2]):
            plt.imsave(f'slices/{name}2\\{i}.png', tensor[:,:,i], cmap='gray');
            plt.close('all');
    
    # save_slices('mri_highlighted_0', tensors[0]);
    # save_slices('mri_highlighted_1', tensors[1]);
    # save_slices('predicted_mri_noisy', tensors[2]);
    if axis == 2 or axis == 1:
        mri_highlighted_0 = [cv2.rotate(cv2.imread(f'slices/mri_highlighted_0{axis}/{i}.png'), cv2.ROTATE_90_COUNTERCLOCKWISE) for i in range(r[0], r[1])];
        mri_highlighted_1 = [cv2.rotate(cv2.cvtColor(cv2.imread(f'slices/mri_highlighted_1{axis}/{i}.png'), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_COUNTERCLOCKWISE) for i in range(r[0], r[1])];
        predicted_mri_noisy = [cv2.rotate(cv2.imread(f'slices/predicted_mri_noisy{axis}/{i}.png'), cv2.ROTATE_90_COUNTERCLOCKWISE) for i in range(r[0], r[1])];
    elif axis == 0:
        mri_highlighted_0 = [cv2.flip(cv2.rotate(cv2.imread(f'slices/mri_highlighted_0{axis}/{i}.png'), cv2.ROTATE_90_COUNTERCLOCKWISE),1) for i in range(r[0], r[1])];
        mri_highlighted_1 = [cv2.flip(cv2.rotate(cv2.cvtColor(cv2.imread(f'slices/mri_highlighted_1{axis}/{i}.png'), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_COUNTERCLOCKWISE),1) for i in range(r[0], r[1])];
        predicted_mri_noisy = [cv2.flip(cv2.rotate(cv2.imread(f'slices/predicted_mri_noisy{axis}/{i}.png'), cv2.ROTATE_90_COUNTERCLOCKWISE),1) for i in range(r[0], r[1])];


    total_concatenated = [];
    for i in range(len(mri_highlighted_0)):
        concatenated = np.zeros((mri_highlighted_0[0].shape[0], mri_highlighted_0[0].shape[1]*3, 3));
        concatenated[:,:mri_highlighted_0[0].shape[1],:] = mri_highlighted_0[i];
        concatenated[:,mri_highlighted_0[0].shape[1]:mri_highlighted_0[0].shape[1]*2,:] = predicted_mri_noisy[i];
        concatenated[:,mri_highlighted_0[0].shape[1]*2:mri_highlighted_0[0].shape[1]*3,:] = mri_highlighted_1[i];
       
        total_concatenated.append(concatenated.astype("uint8"));
    imageio.mimsave('movie.gif', total_concatenated, duration=0.8)

def predict_on_mri_3d(args, model, use_cached = False):
    """predict on mri for detecting all the changes utilizing the self-supervised model

    Parameters
    ----------
    first_mri_path : string
        path to the first MRI
    
    second_mri_path : string
        path to the second MRI
    
    brain_mask_path : string
        path to the brain mask of the MRI
    
    model: nn.Module
        model to use for predictions

    use_cached: bool
        indicate if we are using the previous predictions or should predicition be calculated again
    """

    file_name = os.path.basename(args.sample_time1);
    file_name = file_name[:file_name.find('.')];
    
    if use_cached is False:

        normalize_internsity = NormalizeIntensity(subtrahend=0.5, divisor=0.5);
        def preprocess(mrimage):
            mask = mrimage > threshold_otsu(mrimage);
            mask = np.expand_dims(mask, axis=0);
            mrimage = np.expand_dims(mrimage, axis=0);
            mrimage = mrimage / np.max(mrimage);
            
            return mrimage, mask; 

        with torch.no_grad():

            first_mri_nib = nib.load(args.sample_time1)
            second_mri_nib = nib.load(args.sample_time2)
            brain_mask = nib.load(args.sample_brainmask)

            first_mri_data = first_mri_nib.get_fdata()
            second_mri_data = second_mri_nib.get_fdata()
            brain_mask = brain_mask.get_fdata()

            first_mri_data = window_center_adjustment(first_mri_data);
            second_mri_data = window_center_adjustment(second_mri_data);

            hist = np.histogram(second_mri_data.ravel(), bins = int(np.max(second_mri_data)))[0];
            hist = hist[1:]
            hist = hist / (hist.sum()+1e-4);
            hist = np.cumsum(hist);
            t = np.where(hist<0.5)[0][-1];
            t = t/255;

            first_mri_data, first_mri_data_mask = preprocess(first_mri_data);
            second_mri_data, rigid_registered_image_data_mask = preprocess(second_mri_data);


            mask = second_mri_data > threshold_otsu(second_mri_data); 

            g_inpaint = (second_mri_data > 0.9);
            g_inpaint = torch.from_numpy(g_inpaint);

            t1 =  torch.where(torch.from_numpy(second_mri_data)>t, 0, 1);
            t2 = (second_mri_data > threshold_otsu(second_mri_data));
            g = t1 * t2
            g = g.numpy();
            g = torch.from_numpy(g);

            total_heatmap = torch.zeros_like(torch.from_numpy(second_mri_data), dtype=torch.float64);

            pos_cords = np.where(total_heatmap>0);
            second_mri_data, heatmap, noise, center = inpaint_3d(torch.from_numpy(second_mri_data), g_inpaint, 40, False)

            for _ in range(40): # a lot of added lesions for illustration purposes
                second_mri_data, heatmap = add_synthetic_lesion_wm(second_mri_data, g, False)
                total_heatmap = torch.clamp(heatmap+total_heatmap, 0, 1);
            
            first_mri_data = first_mri_data * mask;
            second_mri_data = second_mri_data * mask;

            first_mri_data = normalize_internsity(first_mri_data)[0];
            second_mri_data = normalize_internsity(second_mri_data)[0];

            first_mri_data = first_mri_data.squeeze();
            second_mri_data = second_mri_data.squeeze();

            w,h,d = first_mri_data.shape;
            new_w = math.ceil(w / args.crop_size_w) * args.crop_size_w;
            new_h = math.ceil(h / args.crop_size_h) * args.crop_size_h;
            new_d = math.ceil(d / args.crop_size_d) * args.crop_size_d;

            first_mri_data_padded  = torch.zeros((new_w, new_h, new_d), dtype = first_mri_data.dtype);
            second_mri_data_padded  = torch.zeros((new_w, new_h, new_d), dtype = first_mri_data.dtype);
            brainmask_data_padded  = torch.zeros((new_w, new_h, new_d), dtype = first_mri_data.dtype);

            first_mri_data_padded[:w,:h,:d] = first_mri_data;
            second_mri_data_padded[:w,:h,:d] = second_mri_data;
            brainmask_data_padded[:w,:h,:d] = torch.from_numpy(brain_mask);

            step_w, step_h, step_d = args.crop_size_w, args.crop_size_h, args.crop_size_d;
            first_mri_data_patches = patchify(first_mri_data_padded.numpy(), 
                                                (args.crop_size_w, args.crop_size_h, args.crop_size_d), 
                                                (step_w, step_h, step_d));
            
            second_mri_data_patches = patchify(second_mri_data_padded.numpy(), 
                                                (args.crop_size_w, args.crop_size_h, args.crop_size_d), 
                                                (step_w, step_h, step_d));
            
            brainmask_data_patches = patchify(brainmask_data_padded.numpy(), 
                                                (args.crop_size_w, args.crop_size_h, args.crop_size_d), 
                                                (step_w, step_h, step_d));

            predicted_mri = np.zeros((new_w, new_h, new_d,3), dtype = np.float64);
            predicted_mri_noisy = np.zeros((new_w, new_h, new_d), dtype = np.float64);
            predicted_positive_thresh = np.zeros((new_w, new_h, new_d,1), dtype = np.float64);
            predicted_negative_thresh = np.zeros((new_w, new_h, new_d,1), dtype = np.float64);
            predicted_hm1_color = np.zeros((new_w, new_h, new_d,4), dtype = np.float64);
            for i in range(first_mri_data_patches.shape[0]):
                for j in range(first_mri_data_patches.shape[1]):
                    for k in range(first_mri_data_patches.shape[2]):
                        
                        first_mri_data_trans = torch.from_numpy(first_mri_data_patches[i,j,k,:,:,:]);
                        second_mri_data_trans = torch.from_numpy(second_mri_data_patches[i,j,k,:,:,:]);
                        brainmask_crop = torch.from_numpy(brainmask_data_patches[i,j,k,:,:,:]);

                        mri, mri_noisy, brainmask_crop = first_mri_data_trans.to(args.device).unsqueeze(dim=0).unsqueeze(dim=0), second_mri_data_trans.to(args.device).unsqueeze(dim=0).unsqueeze(dim=0), brainmask_crop.to(args.device);
                        volume_batch1 = torch.cat([mri, mri_noisy, mri - mri_noisy], dim=1)
                        volume_batch2 = torch.cat([mri_noisy, mri, mri_noisy - mri], dim=1)
                       
                        hm1 = model(volume_batch1);
                        hm2 = model(volume_batch2);

                        hm1 = hm1 * brainmask_crop;
                        hm2 = hm2 * brainmask_crop;

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
                        predicted_mri[i*step_w:(i)*step_w + args.crop_size_w, 
                                    j*step_h:(j)*step_h + args.crop_size_h, 
                                    k*step_d:(k)*step_d + args.crop_size_d,:] = mri;
                        
                        predicted_mri_noisy[i*step_w:(i)*step_w + + args.crop_size_w, 
                                    j*step_h:(j)*step_h+ args.crop_size_h, 
                                    k*step_d:(k)*step_d+ args.crop_size_d] = mri_noisy;
                        
                        predicted_negative_thresh[i*step_w:(i)*step_w + + args.crop_size_w, 
                                    j*step_h:(j)*step_h+ args.crop_size_h, 
                                    k*step_d:(k)*step_d+ args.crop_size_d,:] = hm1_negative_thresh;
                        
                        predicted_positive_thresh[i*step_w:(i)*step_w + + args.crop_size_w, 
                                    j*step_h:(j)*step_h+ args.crop_size_h, 
                                    k*step_d:(k)*step_d+ args.crop_size_d,:] = hm1_positive_thresh;
                        
                        predicted_hm1_color[i*step_w:(i)*step_w + + args.crop_size_w, 
                                    j*step_h:(j)*step_h+ args.crop_size_h, 
                                    k*step_d:(k)*step_d+ args.crop_size_d,:] = hm1_color;

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

    fig,ax = plt.subplots(3,3);  

    global x,y,z;
    x = 10;
    y = 10;
    z = 10;
    intensity_scale = 0.0;

    global mri_highlighted;
    mri_highlighted = (1-predicted_negative_thresh)*predicted_mri + (predicted_negative_thresh * (predicted_hm1_color[:,:,:,:3]*intensity_scale + predicted_mri*(1-intensity_scale)));
    mri_highlighted = (1-predicted_positive_thresh)*mri_highlighted + (predicted_positive_thresh * (predicted_hm1_color[:,:,:,:3]*intensity_scale + mri_highlighted*(1-intensity_scale)));
    mri_highlighted_0 = copy(mri_highlighted);
    
    intensity_scale = 1.0;
    mri_highlighted = (1-predicted_negative_thresh)*predicted_mri + (predicted_negative_thresh * (predicted_hm1_color[:,:,:,:3]*intensity_scale + predicted_mri*(1-intensity_scale)));
    mri_highlighted = (1-predicted_positive_thresh)*mri_highlighted + (predicted_positive_thresh * (predicted_hm1_color[:,:,:,:3]*intensity_scale + mri_highlighted*(1-intensity_scale)));
    mri_highlighted_1 = copy(mri_highlighted);

    ax[0][0].imshow(mri_highlighted[x, :, :], cmap='hot');
    ax[0][1].imshow(mri_highlighted[:,y, :], cmap='hot');
    ax[0][2].imshow(mri_highlighted[ :, :,z], cmap='hot');
    ax[0][0].axis('off');
    ax[0][1].axis('off');
    ax[0][2].axis('off');


    ax[1][0].imshow(predicted_mri_noisy[x, :, :], cmap='gray');
    ax[1][1].imshow(predicted_mri_noisy[:, y, :], cmap='gray');
    ax[1][2].imshow(predicted_mri_noisy[ :, :,z], cmap='gray');
    ax[1][0].axis('off');
    ax[1][1].axis('off');
    ax[1][2].axis('off');

    
    ax[2][0].imshow((predicted_negative_thresh[x, :, :] + predicted_positive_thresh[x, :, :]) * predicted_hm1_color[x, :, :,:3], cmap='hot');
    ax[2][1].imshow((predicted_negative_thresh[:, y, :] + predicted_positive_thresh[:, y, :]) * predicted_hm1_color[:,y, :,:3], cmap='hot');
    ax[2][2].imshow((predicted_negative_thresh[ :, :,z] + predicted_positive_thresh[:, :, z]) * predicted_hm1_color[ :, :,z,:3], cmap='hot');
    ax[2][0].axis('off');
    ax[2][1].axis('off');
    ax[2][2].axis('off');

    if args.gif:
        save_to_gif([mri_highlighted_0, mri_highlighted_1, predicted_mri_noisy, predicted_negative_thresh, predicted_positive_thresh, predicted_hm1_color], 
                    0,
                    [120, 180]);
    else:

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
        
            ax[2][0].imshow((predicted_negative_thresh[x, :, :] + predicted_positive_thresh[x, :, :]) * predicted_hm1_color[x, :, :,:3], cmap='hot');
            ax[2][1].imshow((predicted_negative_thresh[:, y, :] + predicted_positive_thresh[:, y, :]) * predicted_hm1_color[:,y, :,:3], cmap='hot');
            ax[2][2].imshow((predicted_negative_thresh[ :, :,z] + predicted_positive_thresh[:, :, z]) * predicted_hm1_color[ :, :,z,:3], cmap='hot');

    

        def update_slice_y(val):
            global y;
            y = int(val);
            ax[0][0].imshow(mri_highlighted[x, :, :]);
            ax[0][1].imshow(mri_highlighted[:,y, :]);
            ax[0][2].imshow(mri_highlighted[ :, :,z]);

            ax[1][0].imshow(predicted_mri_noisy[x, :, :], cmap='gray');
            ax[1][1].imshow(predicted_mri_noisy[:,y, :], cmap='gray');
            ax[1][2].imshow(predicted_mri_noisy[ :, :,z], cmap='gray');
        
            ax[2][0].imshow((predicted_negative_thresh[x, :, :] + predicted_positive_thresh[x, :, :]) * predicted_hm1_color[x, :, :,:3], cmap='hot');
            ax[2][1].imshow((predicted_negative_thresh[:, y, :] + predicted_positive_thresh[:, y, :]) * predicted_hm1_color[:,y, :,:3], cmap='hot');
            ax[2][2].imshow((predicted_negative_thresh[ :, :,z] + predicted_positive_thresh[:, :, z]) * predicted_hm1_color[ :, :,z,:3], cmap='hot');

        
        def update_slice_z(val):
            global z;
            z = int(val);
            ax[0][0].imshow(mri_highlighted[x, :, :]);
            ax[0][1].imshow(mri_highlighted[:,y, :]);
            ax[0][2].imshow(mri_highlighted[ :, :,z]);

            ax[1][0].imshow(predicted_mri_noisy[x, :, :], cmap='gray');
            ax[1][1].imshow(predicted_mri_noisy[:,y, :], cmap='gray');
            ax[1][2].imshow(predicted_mri_noisy[ :, :,z], cmap='gray');
        
            ax[2][0].imshow((predicted_negative_thresh[x, :, :] + predicted_positive_thresh[x, :, :]) * predicted_hm1_color[x, :, :,:3], cmap='hot');
            ax[2][1].imshow((predicted_negative_thresh[:, y, :] + predicted_positive_thresh[:, y, :]) * predicted_hm1_color[:,y, :,:3], cmap='hot');
            ax[2][2].imshow((predicted_negative_thresh[ :, :,z] + predicted_positive_thresh[:, :, z]) * predicted_hm1_color[ :, :,z,:3], cmap='hot');


        amp_slider.on_changed(update)
        x_slider.on_changed(update_slice_x)
        y_slider.on_changed(update_slice_y)
        z_slider.on_changed(update_slice_z)
        plt.show();

def parse_file():
    """parse results file for 5 folds and print average along with std in parantheses
    """
    total_dice = [];
    total_hd = [];
    total_f1 = [];
    for f in range(0,5):
        with open(os.path.join('Results',f'f{f}.txt'), 'r') as fil:
            fil.readline();
            dice = fil.readline().rstrip();
            dice = float(dice[dice.rfind(':')+1:]);
            hd = fil.readline().rstrip();
            hd = float(hd[hd.rfind(':')+1:]);
            f1 = fil.readline().rstrip();
            f1 = float(f1[f1.rfind(':')+1:]);
            total_dice.append(dice);
            total_hd.append(hd);
            total_f1.append(f1);

    print(f'Dice: {np.mean(total_dice)} ({np.std(total_dice)})\tHD: {np.mean(total_hd)}({np.std(total_hd)})\tF1: {np.mean(total_f1)}({np.std(total_f1)})');

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SSLMRI Testing', allow_abbrev=False);
    parser.add_argument('--segmentation', default=True, action='store_false', help='if true calculate results for segmentation, if false, shows an example of detecting all changes between two MRI scans')
    parser.add_argument('--model-path', type=str, default='models/', help='path to pretrained models, add them beside code to "models" folders, all five models should be in seperate folders named f0,f1,f2,f3,f4')
    parser.add_argument('--device', type=str, default='cuda', help='device to run model on');
    parser.add_argument('--gif', default=False, action='store_true', help='indicate to create a gif from output or show it using matplotlib, only for pretrained model');
    parser.add_argument('--use-cached', default=False, action='store_true', help='if true, used already cached predictions, if false, do prediction first before showing results');
    parser.add_argument('--sample-time1', type = str, default='samples/time1.nii.gz',help = 'path to first time point for regressing all changes');
    parser.add_argument('--sample-time2', type = str, default='samples/time2.nii.gz',help = 'path to first time point for regressing all changes');
    parser.add_argument('--sample-brainmask', type = str, default='samples/brainmask.nii.gz',help = 'path to brain mask for regressing all changes');
    parser.add_argument('--batch-size', default=4, type=int);
    parser.add_argument('--crop-size-w', default=96, type=int, help='crop size for getting a patch from MRI scan');
    parser.add_argument('--crop-size-h', default=96, type=int, help='crop size for getting a patch from MRI scan');
    parser.add_argument('--crop-size-d', default=96, type=int, help='crop size for getting a patch from MRI scan');
    parser.add_argument('--learning-rate', default=1e-4, type=float);
    parser.add_argument('--sample-per-mri', default=8, type=int, help='how many samples to take from each MRI scan');
    parser.add_argument('--deterministic', default=False, action='store_true', help='if we want to have same augmentation and same datae, for sanity check');
    parser.add_argument('--virtual-batch-size', default=1, type=int, help='use it if batch size does not fit GPU memory');
    parser.add_argument('--num-workers', default=0, type=int, help='num workers for data loader, should be equal to number of CPU cores');
    parser.add_argument('--bl-multiplier', default=10, type=int, help='boundary loss coefficient');
    parser.add_argument('--debug-train-data', default=False, action='store_true', help='debug training data for debugging purposes');
    parser.add_argument('--pretrained', default=False, action='store_true', help='indicate wether to initalize with self-supervised pretrained  model or not');
    parser.add_argument('--pretraining', default=True, action='store_true', help='indicate if we are doing self-supervised pretraining (True) or training of segmenation model (False)');
    parser.add_argument('--fold', default=0, type=int, help='which fold to train and test model on');
    parser.add_argument('--network', default='VNET', type=str, help='which model to use, SWINUNETR is the other option');
    parser.add_argument('--pretrain-path', default=f'best_model.ckpt', type=str, help='path to self-supervised pretrain model');
    parser.add_argument('--resume', default=False, action='store_true',  help='inidcate wether we are training from scratch or resume training');
    parser.add_argument('--cache-mri-data', default=False, action='store_true',  help='if true, it first generate testing set for self-supervised pretraining mode, should run only once');
    parser.add_argument('--num-cache-data', default=200,  help='number of examples to cache for testing of self-supervised pretraining');
    
    args = parser.parse_args();

    if args.segmentation is True:
        model = VNet('segmentation', n_channels=3, n_classes=1, normalization='batchnorm', has_dropout=True)
    else:
        model = VNet('pretraining', n_channels=3, n_classes=1, normalization='batchnorm', has_dropout=True)

    if args.segmentation is True:
       
        for f in range(0,5):
            ckpt = torch.load(os.path.join(args.model_path, f'f{f}',  'best_model.ckpt'));
            model.load_state_dict(ckpt['model']);
            model.to(args.device);
            model.eval();

            train_loader, test_ids, test_dataset = get_loader_miccai(args, f);
            
            valid_res = valid(model, test_ids, test_dataset);
            if os.path.exists(os.path.join('Results')) is False:
                os.makedirs(os.path.join('Results'));
            
            with open(os.path.join('Results', f'f{f}.txt'), 'w') as fil:
                fil.write(f'f{f}');
                fil.write("\n");
                fil.write(f'dice: {valid_res[0]}');
                fil.write("\n");
                fil.write(f'hd: {valid_res[1]}');
                fil.write("\n");
                fil.write(f'f1: {valid_res[2]}');
                fil.close();
        
        parse_file()

    else:
        ckpt = torch.load(os.path.join(args.model_path, 'pretraining', 'best_model.ckpt'));
        model.load_state_dict(ckpt['model']);
        model.to(args.device);
        model.eval();
        predict_on_mri_3d(args, model);
    
