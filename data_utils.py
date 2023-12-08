import os
from torch.utils.data import DataLoader, Dataset
import pickle
from glob import glob
import numpy as np
from copy import deepcopy, copy
import matplotlib.pyplot as plt

import torch
import config
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import nibabel as nib
from skimage.filters import gaussian, sobel, threshold_otsu, try_all_threshold, threshold_triangle, threshold_mean
from monai.transforms import Compose, Resize, SpatialPadd, ScaleIntensityRange, Rand3DElastic, Resize, RandGaussianSmooth, OneOf, RandGibbsNoise, RandGaussianNoise, GaussianSmooth, NormalizeIntensity, RandCropByPosNegLabeld, GibbsNoise, RandSpatialCropSamplesd
from scipy.ndimage import convolve, binary_erosion, binary_opening
from tqdm import tqdm
import math
from patchify import patchify
import seaborn as sns
from scipy.ndimage import distance_transform_edt, sobel, histogram, prewitt,laplace, gaussian_filter
from monai.losses.dice import DiceLoss
from utility import calculate_metric_percase


def window_center_adjustment(img):

    hist = np.histogram(img.ravel(), bins = int(np.max(img)))[0];
    hist = hist / (hist.sum()+1e-4);
    hist = np.cumsum(hist);

    hist_thresh = ((1-hist) < 5e-4);
    max_intensity = np.where(hist_thresh == True)[0][0];
    adjusted_img = img * (255/(max_intensity + 1e-4));
    adjusted_img = np.where(adjusted_img > 255, 255, adjusted_img).astype("uint8");

    return adjusted_img;

def cache_test_dataset(num_data, test_mri, fold):
    if os.path.exists(f'cache/{fold}') is False:
        os.makedirs(f'cache/{fold}');
    
    mri_dataset_test = MRI_Dataset(test_mri,cache=True);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=0, pin_memory=True);

    num_data = math.ceil(num_data/len(test_loader));
    counter = 0;
    ret = [];
    for n in range(num_data):
        for (batch) in test_loader:
            ret.append(os.path.join('cache',f'{fold}', f'{counter}.tstd'))
            pickle.dump([b.squeeze() for b in batch], open(os.path.join('cache',f'{fold}', f'{counter}.tstd'), 'wb'));
            counter += 1;

    return ret;


def cache_test_dataset_miccai(num_data, fold):
    if os.path.exists(f'cache_miccai/{fold}') is False:
        os.makedirs(f'cache_miccai/{fold}');
    
    with open(os.path.join('cache_miccai', f'fold{fold}.txt'), 'r') as f:
        train_ids = f.readline().rstrip();
        train_ids = train_ids.split(',');
        test_ids = f.readline().rstrip();
        test_ids = test_ids.split(',');
    
    test_ids = [os.path.join('miccai-processed', t) for t in test_ids];
    mri_paths = [];
    for t in test_ids:
        mri_paths.append(os.path.join(t, 'flair_time01_on_middle_space.nii.gz'));
        mri_paths.append(os.path.join(t, 'flair_time02_on_middle_space.nii.gz'));

    
    mri_dataset_test = MICCAI_PRETRAIN_Dataset(mri_paths,cache=True);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=0, pin_memory=True);

    num_data = math.ceil(num_data/len(test_loader));
    counter = 0;
    ret = [];
    for n in tqdm(range(num_data)):
        for (batch) in test_loader:
            ret.append(os.path.join('cache_miccai',f'{fold}', f'{counter}.tstd'))
            pickle.dump([b.squeeze() for b in batch], open(os.path.join('cache_miccai',f'{fold}', f'{counter}.tstd'), 'wb'));
            counter += 1;

    return ret;

class MRI_Dataset(Dataset):
    def __init__(self, mr_images, train = True, cache = False) -> None:
        super().__init__();
        m1 = 0.7;
        m2 = 0.8;
        self.augment_noisy_image = OneOf([
            RandGaussianSmooth(prob=1.0, sigma_x=(m1,m2), sigma_y=(m1,m2), sigma_z=(m1,m2)),
            RandGaussianNoise(prob=1.0,std=0.05),
            RandGibbsNoise(prob=1.0, alpha=(0.65,0.75))
        ], weights=[1,1,1])


        self.transforms = NormalizeIntensity(subtrahend=0.5, divisor=0.5);
        self.crop = RandCropByPosNegLabeld(
            keys=['image', 'gradient', 'mask'], 
            label_key='mask', 
            spatial_size= (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']),
            pos=1, 
            neg=0,
            num_samples=1 if cache else config.hyperparameters['sample_per_mri'] if train else 1);
        self.resize = Resize(spatial_size=[config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']]);

        self.train = train;
        self.cache = cache;

        self.mr_images = [];

        if train or cache:
            for mr in mr_images:
                file_name = mr[mr.rfind('\\')+1:];
                file_name = file_name[:file_name.find('.')];
                g = pickle.load(open(os.path.join('mri_data', f'{file_name}.gradient'), 'rb'));
                g = np.expand_dims(g, axis=0);

                mrimage = nib.load(mr.replace('\\','/'))
                mrimage = nib.as_closest_canonical(mrimage);
                mrimage = mrimage.get_fdata()
                mrimage = window_center_adjustment(mrimage);
                self.mr_images.append([mrimage, g]);
        else:
            for mr in mr_images:
                mrimage = pickle.load(open(mr, 'rb'))
                self.mr_images.append(mrimage);
    
    def __len__(self):
        return len(self.mr_images);

    def __getitem__(self, index):
        if self.train:
            # mrimage = nib.load(self.mr_images[index])
            # mrimage = nib.as_closest_canonical(mrimage);
            # mrimage = mrimage.get_fdata()
            # mrimage = window_center_adjustment(mrimage);
            mrimage = self.mr_images[index][0];
            mask = mrimage > threshold_otsu(mrimage);
            g = self.mr_images[index][1];
            
            mask = np.expand_dims(mask, axis=0);
            mrimage = np.expand_dims(mrimage, axis=0);
            mrimage = mrimage / (np.max(mrimage)+1e-4);
            

            ret_mrimage = None;
            ret_mrimage_noisy = None;
            ret_mask = None;
            ret_total_heatmap = None;

            if config.hyperparameters['deterministic'] is False:
                ret_transforms = self.crop({'image': mrimage,'gradient': g, 'mask': mask});
            
            for i in range(config.hyperparameters['sample_per_mri'] if self.cache is False else 1):
                if config.hyperparameters['deterministic'] is False:
                    mrimage_c = ret_transforms[i]['image'];
                    g_c = ret_transforms[i]['gradient'];
                    mask_c = ret_transforms[i]['mask'];
                    mrimage_noisy = copy(mrimage_c);
                else:
                    mrimage_c = mrimage[:, int(mrimage.shape[1]/2-32):int(mrimage.shape[1]/2+32), int(mrimage.shape[2]/2-32):int(mrimage.shape[2]/2+32), int(mrimage.shape[3]/2-32):int(mrimage.shape[3]/2+32)];
                    g_c = g[:, int(mrimage.shape[1]/2-32):int(mrimage.shape[1]/2+32), int(mrimage.shape[2]/2-32):int(mrimage.shape[2]/2+32), int(mrimage.shape[3]/2-32):int(mrimage.shape[3]/2+32)];
                    mask_c = mrimage[:, int(mrimage.shape[1]/2-32):int(mrimage.shape[1]/2+32), int(mrimage.shape[2]/2-32):int(mrimage.shape[2]/2+32), int(mrimage.shape[3]/2-32):int(mrimage.shape[3]/2+32)];
                    mrimage_noisy = copy(mrimage_c);
                    mrimage_c = torch.from_numpy(mrimage_c);
                    mrimage_noisy = torch.from_numpy(mrimage_noisy);
                    g_c = torch.from_numpy(g_c);
                    mask_c = torch.from_numpy(mask_c);

                total_heatmap = torch.zeros_like(mrimage_noisy, dtype=torch.float64);


                num_corrupted_patches = np.random.randint(1,5) if config.hyperparameters['deterministic'] is False else 3;
                for i in range(num_corrupted_patches):
                    mrimage_noisy, heatmap, noise, center = inpaint_3d(mrimage_noisy, g_c)
                    total_heatmap += heatmap;
                # total_heatmap_thresh = total_heatmap > 0;
                #pos_cords = np.where(total_heatmap_thresh == 1);
                #r = np.random.randint(0,len(pos_cords[0]));
                #center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
                total_heatmap_thresh = torch.where(total_heatmap > 0.9, 0.0, 1.0);
                if config.hyperparameters['deterministic'] is True:
                    mrimage_noisy = GibbsNoise(alpha = 0.65)(mrimage_noisy);
                else:
                    mrimage_noisy = self.augment_noisy_image(mrimage_noisy);
                #visualize_2d([mrimage_c, mrimage_noisy, total_heatmap, noise], center);
                
                
                mrimage_c = self.transforms(mrimage_c)[0];
                mrimage_noisy = self.transforms(mrimage_noisy)[0];

                if ret_mrimage is None:
                    ret_mrimage = mrimage_c.unsqueeze(dim=0);
                    ret_mrimage_noisy = mrimage_noisy.unsqueeze(dim=0);
                    ret_mask = mask_c.unsqueeze(dim=0);
                    ret_total_heatmap = total_heatmap_thresh.unsqueeze(dim=0);
                else:
                    ret_mrimage = torch.concat([ret_mrimage, mrimage_c.unsqueeze(dim=0)], dim=0);
                    ret_mrimage_noisy = torch.concat([ret_mrimage_noisy, mrimage_noisy.unsqueeze(dim=0)], dim=0);
                    ret_mask = torch.concat([ret_mask, mask_c.unsqueeze(dim=0)], dim=0);
                    ret_total_heatmap = torch.concat([ret_total_heatmap, total_heatmap_thresh.unsqueeze(dim=0)], dim=0);
            
            return ret_mrimage, ret_mrimage_noisy, ret_mask, ret_total_heatmap;
        else:
            ret = self.mr_images[index];
            return ret;

def cropper(mri1, mri2, gt, gr, roi_size, num_samples):
    ret = [];
    for i in range(num_samples):
        pos_cords = np.where(gt > 0);
        r = np.random.randint(0,len(pos_cords[0]));
        center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
        d_x_l = min(roi_size[0]//2,center[0]);
        d_x_r = min(roi_size[0]//2 ,mri1.shape[1]-center[0]);
        if d_x_l != roi_size[0]//2:
            diff = abs(roi_size[0]//2 - center[0]);
            d_x_r += diff;
        if d_x_r != roi_size[0]//2 and d_x_l == roi_size[0]//2:
            diff = abs(roi_size[0]//2 - (mri1.shape[1]-center[0]));
            d_x_l += diff;
        
        d_y_l = min(roi_size[1]//2,center[1]);
        d_y_r = min(roi_size[1]//2 ,mri1.shape[2]-center[1]);
        if d_y_l != roi_size[1]//2:
            diff = abs(roi_size[1]//2 - center[1]);
            d_y_r += diff;
        if d_y_r != roi_size[1]//2 and d_y_l == roi_size[1]//2:
            diff = abs(roi_size[1]//2 - mri1.shape[2]-center[1]);
            d_y_l += diff;
        
        d_z_l = min(roi_size[2]//2,center[2]);
        d_z_r = min(roi_size[2]//2 ,mri1.shape[3]-center[2]);
        if d_z_l != roi_size[2]//2:
            diff = abs(roi_size[2]//2 - center[2]);
            d_z_r += diff;
        if d_z_r != roi_size[2]//2 and d_z_l == roi_size[2]//2:
            diff = abs(roi_size[2]//2 - mri1.shape[3]-center[2]);
            d_z_l += diff;

        sign_x = np.random.randint(1,3);
        if sign_x%2!=0:
            offset_x = np.random.randint(0, max(min(abs(center[0]-int(d_x_l)), int(d_x_l//2)),1))*-1;
        else:
            offset_x = np.random.randint(0, max(min(abs(center[0]+int(d_x_r)-mri1.shape[1]), int(d_x_r//2)), 1));
        start_x = center[0]-int(d_x_l)+offset_x;
        end_x = center[0]+int(d_x_r)+offset_x;

        sign_y = np.random.randint(1,3);
        if sign_y%2!=0:
            offset_y = np.random.randint(0, max(min(abs(center[1]-int(d_y_l)), int(d_y_l//2)),1))*-1;
        else:
            offset_y = np.random.randint(0, max(min(abs(center[1]+int(d_y_r)-mri1.shape[2]), int(d_y_r//2)), 1));
        start_y = center[1]-int(d_y_l) + offset_y;
        end_y = center[1]+int(d_y_r) + offset_y;

        sign_z = np.random.randint(1,3);
        if sign_z%2!=0:
            offset_z = np.random.randint(0, max(min(abs(center[2]-int(d_z_l)), int(d_z_l)),1))*-1;
        else:
            offset_z = np.random.randint(0, max(min(abs(center[2]+int(d_z_r)-mri1.shape[3]), int(d_z_r//2)), 1));
        
        start_z = center[2]-int(d_z_l)+offset_z;
        end_z = center[2]+int(d_z_r)+offset_z;

        d = dict();
        d['image1'] = torch.from_numpy(mri1[:, start_x:end_x, start_y:end_y, start_z:end_z]);
        d['image2'] = torch.from_numpy(mri2[:, start_x:end_x, start_y:end_y, start_z:end_z]);
        d['mask'] = torch.from_numpy(gt[:, start_x:end_x, start_y:end_y, start_z:end_z]);
        d['gradient'] = torch.from_numpy(gr[:,start_x:end_x, start_y:end_y, start_z:end_z]);

        ret.append(d);

    return ret;

class MICCAI_PRETRAIN_Dataset(Dataset):
    def __init__(self, mr_images, train = True, cache = False) -> None:
        super().__init__();
        m1 = 0.7;
        m2 = 0.8;
        self.augment_noisy_image = OneOf([
            RandGaussianSmooth(prob=1.0, sigma_x=(m1,m2), sigma_y=(m1,m2), sigma_z=(m1,m2)),
            RandGaussianNoise(prob=1.0,std=0.05),
           # RandGibbsNoise(prob=1.0, alpha=(0.65,0.75))
        ], weights=[1,1])


        self.transforms = NormalizeIntensity(subtrahend=0.5, divisor=0.5);
        self.crop = RandCropByPosNegLabeld(
            keys=['image', 'gradient', 'thresh', 'mask'], 
            label_key='mask', 
            spatial_size= (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']),
            pos=1, 
            neg=0,
            num_samples=1 if cache else config.hyperparameters['sample_per_mri'] if train else 1);
        self.resize = Resize(spatial_size=[config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']]);

        self.train = train;
        self.cache = cache;

        self.mr_images = [];

        if train or cache:
            for patient_path in mr_images:

                mri = nib.load(patient_path);
                gradient_file_path = patient_path[:patient_path.find('.')];
                mri = mri.get_fdata();
                mri = window_center_adjustment(mri);

                gradient = pickle.load(open(f'{gradient_file_path}.gradient', 'rb'));
                self.mr_images.append([mri, gradient])

        else:
            for mr in mr_images:
                mrimage = pickle.load(open(mr, 'rb'))
                self.mr_images.append(mrimage);
    
    def __len__(self):
        return len(self.mr_images);

    def __getitem__(self, index):
        if self.train:
            # mrimage = nib.load(self.mr_images[index])
            # mrimage = nib.as_closest_canonical(mrimage);
            # mrimage = mrimage.get_fdata()
            # mrimage = window_center_adjustment(mrimage);
            mrimage, gr = self.mr_images[index];
            mask = mrimage > threshold_otsu(mrimage);            
            mask = np.expand_dims(mask, axis=0);
            mrimage = np.expand_dims(mrimage, axis=0);
            mrimage = mrimage / (np.max(mrimage)+1e-4);

           # gr = sobel(mrimage);
            g = (mrimage > 0.9) * gr;
            g = binary_opening(g.squeeze(), structure=np.ones((2,2,2))).astype(g.dtype)
            g = torch.from_numpy(np.expand_dims(g, axis=0));
            #g = g < threshold_otsu(g);
           # g = np.expand_dims(gr, axis=0);
            

            ret_mrimage = None;
            ret_mrimage_noisy = None;
            ret_mask = None;
            ret_total_heatmap = None;

            if config.hyperparameters['deterministic'] is False:
                ret_transforms = self.crop({'image': mrimage,'gradient': g, 'thresh': mask, 'mask': g});
            
            for i in range(config.hyperparameters['sample_per_mri'] if self.cache is False else 1):
                if config.hyperparameters['deterministic'] is False:
                    mrimage_c = ret_transforms[i]['image'];
                    g_c = ret_transforms[i]['gradient'];
                    mask_c = ret_transforms[i]['thresh'];
                    mrimage_noisy = copy(mrimage_c);
                else:
                    mrimage_c = mrimage[:, int(mrimage.shape[1]/2-32):int(mrimage.shape[1]/2+32), int(mrimage.shape[2]/2-32):int(mrimage.shape[2]/2+32), int(mrimage.shape[3]/2-32):int(mrimage.shape[3]/2+32)];
                    g_c = g[:, int(mrimage.shape[1]/2-32):int(mrimage.shape[1]/2+32), int(mrimage.shape[2]/2-32):int(mrimage.shape[2]/2+32), int(mrimage.shape[3]/2-32):int(mrimage.shape[3]/2+32)];
                    mask_c = mrimage[:, int(mrimage.shape[1]/2-32):int(mrimage.shape[1]/2+32), int(mrimage.shape[2]/2-32):int(mrimage.shape[2]/2+32), int(mrimage.shape[3]/2-32):int(mrimage.shape[3]/2+32)];
                    mrimage_noisy = copy(mrimage_c);
                    mrimage_c = torch.from_numpy(mrimage_c);
                    mrimage_noisy = torch.from_numpy(mrimage_noisy);
                    g_c = torch.from_numpy(g_c);
                    mask_c = torch.from_numpy(mask_c);

                total_heatmap = torch.zeros_like(mrimage_noisy, dtype=torch.float64);


                num_corrupted_patches = np.random.randint(1,5) if config.hyperparameters['deterministic'] is False else 3;
                #for i in range(1):
                mrimage_noisy, heatmap, noise, center = inpaint_3d(mrimage_noisy, g_c, num_corrupted_patches)
                total_heatmap += heatmap;
                
                total_heatmap = total_heatmap * mask_c;
                mrimage_noisy = mrimage_noisy * mask_c;
                mrimage_c = mrimage_c * mask_c;
                # total_heatmap_thresh = total_heatmap > 0;
                #pos_cords = np.where(total_heatmap_thresh == 1);
                #r = np.random.randint(0,len(pos_cords[0]));
                #center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
                total_heatmap_thresh = torch.where(total_heatmap > 0.8, 1.0, 0.0);
                part_first = mrimage_c * total_heatmap_thresh;
                part_second = mrimage_noisy * total_heatmap_thresh;
                if config.hyperparameters['deterministic'] is True:
                    mrimage_noisy = GibbsNoise(alpha = 0.65)(mrimage_noisy);
                else:
                    mrimage_noisy = self.augment_noisy_image(mrimage_noisy);

                diff = torch.abs(part_first - part_second) > 0.2;

                
                total_heatmap_thresh = torch.where(diff > 0, 0, 1);
                if config.DEBUG_TRAIN_DATA:
                    visualize_2d([mrimage_c, mrimage_noisy, total_heatmap, diff], center);
                
                
                mrimage_c = self.transforms(mrimage_c)[0];
                mrimage_noisy = self.transforms(mrimage_noisy)[0];

                if ret_mrimage is None:
                    ret_mrimage = mrimage_c.unsqueeze(dim=0);
                    ret_mrimage_noisy = mrimage_noisy.unsqueeze(dim=0);
                    #ret_mask = mask_c.unsqueeze(dim=0);
                    ret_total_heatmap = total_heatmap_thresh.unsqueeze(dim=0);
                else:
                    ret_mrimage = torch.concat([ret_mrimage, mrimage_c.unsqueeze(dim=0)], dim=0);
                    ret_mrimage_noisy = torch.concat([ret_mrimage_noisy, mrimage_noisy.unsqueeze(dim=0)], dim=0);
                    #ret_mask = torch.concat([ret_mask, mask_c.unsqueeze(dim=0)], dim=0);
                    ret_total_heatmap = torch.concat([ret_total_heatmap, total_heatmap_thresh.unsqueeze(dim=0)], dim=0);
            
            return ret_mrimage, ret_mrimage_noisy, ret_total_heatmap;
        else:
            ret = self.mr_images[index];
            return ret;

class MICCAI_Dataset(Dataset):
    def __init__(self, patient_ids, train = True) -> None:
        super().__init__();
        m1 = 0.4;
        m2 = 0.5;
        self.augment_noisy_image = OneOf([
            RandGaussianSmooth(prob=1.0, sigma_x=(m1,m2), sigma_y=(m1,m2), sigma_z=(m1,m2)),
            RandGaussianNoise(prob=1.0,std=.05),
            RandGibbsNoise(prob=1.0, alpha=(0.35,0.45))
        ], weights=[1,1,1])


        self.transforms = Compose(
            [
                
                NormalizeIntensity(subtrahend=0.5, divisor=0.5)
            ]
        )

        self.crop_rand = Compose(
            [ 
                RandCropByPosNegLabeld(
                keys=['image1', 'image2', 'mask', 'lbl', 'gradient'], 
                label_key='image1', 
                spatial_size= (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']),
                pos=1, 
                neg=0,
                num_samples=config.hyperparameters['sample_per_mri'] if train else 1,)
            ]
        )

        self.train = train;

        self.data = [];
        self.pred_data = dict();
        self.gt_data = dict();
        
        if train:
            for patient_path in patient_ids:
                mri1 = nib.load(os.path.join(patient_path, f'flair_time01_on_middle_space.nii.gz'));
                mri1 = mri1.get_fdata();
                mri1 = window_center_adjustment(mri1);

                mri2 = nib.load(os.path.join(patient_path, f'flair_time02_on_middle_space.nii.gz'));
                mri2 = mri2.get_fdata();
                mri2 = window_center_adjustment(mri2);

                gt = nib.load(os.path.join(patient_path, f'ground_truth.nii.gz'));
                gt = gt.get_fdata();

                brainmask = nib.load(os.path.join(patient_path, f'brain_mask.nii.gz'));
                brainmask = brainmask.get_fdata();

                gradient = pickle.load(open(os.path.join(patient_path,'gradient.gradient'), 'rb'));

                gt = gt * brainmask;
                
                self.data.append([mri1, mri2, gt, np.expand_dims(gradient.astype("uint8"),axis=0), patient_path]);
        
        else:
            for patient_path in patient_ids:
                patient_id = patient_path[patient_path.rfind('/')+1:]
                                
                mri1= nib.load(os.path.join(patient_path, f'flair_time01_on_middle_space.nii.gz'));
                mri1 = mri1.get_fdata();
                mri1 = window_center_adjustment(mri1);

                mri2= nib.load(os.path.join(patient_path, f'flair_time02_on_middle_space.nii.gz'));
                mri2 = mri2.get_fdata();
                mri2 = window_center_adjustment(mri2);

                gt = nib.load(os.path.join(patient_path, f'ground_truth.nii.gz'));
                gt = gt.get_fdata();

                brainmask = nib.load(os.path.join(patient_path, f'brain_mask.nii.gz'));
                brainmask = brainmask.get_fdata();
                n = np.max(brainmask);

                gt = gt * brainmask;


                mri1 = mri1 / (np.max(mri1)+1e-4);
                mri2 = mri2 / (np.max(mri2)+1e-4);

                #visualize_2d([mri1,mri2, mask], [20, 20, 20]);


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

                self.step_w, self.step_h, self.step_d = config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d'];
                mri1_patches = patchify(mri1_padded, 
                                                    (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                    (self.step_w, self.step_h, self.step_d));
                mri2_patches = patchify(mri2_padded, 
                                                    (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                    (self.step_w, self.step_h, self.step_d));
                gt_patches = patchify(gt_padded, 
                                                    (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                    (self.step_w, self.step_h, self.step_d));
                brainmask_patches = patchify(brainmask_padded, 
                                                    (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                    (self.step_w, self.step_h, self.step_d));
                # mri1_patches = mri1_patches.reshape(mri1_patches.shape[0]*mri1_patches.shape[1]*mri1_patches.shape[2], mri1_patches.shape[3], mri1_patches.shape[4],mri1_patches.shape[5]);
                # mri2_patches = mri2_patches.reshape(mri2_patches.shape[0]*mri2_patches.shape[1]*mri2_patches.shape[2], mri2_patches.shape[3], mri2_patches.shape[4],mri2_patches.shape[5]);
                # gt_patches = gt_patches.reshape(gt_patches.shape[0]*gt_patches.shape[1]*gt_patches.shape[2], gt_patches.shape[3], gt_patches.shape[4],gt_patches.shape[5]);
                # brainmask_patches = brainmask_patches.reshape(brainmask_patches.shape[0]*brainmask_patches.shape[1]*brainmask_patches.shape[2], brainmask_patches.shape[3], brainmask_patches.shape[4],brainmask_patches.shape[5]);

                curr_data = [];
                for i in range(mri1_patches.shape[0]):
                    for j in range(mri1_patches.shape[1]):
                        for k in range(mri1_patches.shape[2]):
                            curr_data.append((mri1_patches[i,j,k,...],mri2_patches[i,j,k,...],gt_patches[i,j,k,...], patient_id[patient_id.rfind('\\')+1:], brainmask_patches[i,j,k,...], [i,j,k]))

                predicted_aggregated = np.zeros((new_w, new_h, new_d), dtype = np.int32);
                self.pred_data[patient_id[patient_id.rfind('\\')+1:]] = predicted_aggregated;
                self.gt_data[patient_id[patient_id.rfind('\\')+1:]] = gt_padded;

                self.data.extend(curr_data);

            
    
    def __len__(self):
        return len(self.data);

    def __getitem__(self, index):
        if self.train:
            
            mri1, mri2, gt, gr, pp = self.data[index];

            mri1 = np.expand_dims(mri1, axis=0);
            mri2 = np.expand_dims(mri2, axis=0);
            gt = np.expand_dims(gt, axis=0);

            hist = np.histogram(mri2.ravel(), bins = int(np.max(mri2)))[0];
            hist = hist[1:]
            hist = hist / (hist.sum()+1e-4);
            hist = np.cumsum(hist);
            t = np.where(hist<0.5)[0][-1];
            t = t/255;

            

            mri1 = mri1 / (np.max(mri1)+1e-4);
            mri2 = mri2 / (np.max(mri2)+1e-4);
            

            if config.hyperparameters['deterministic'] is False:
                if np.sum(gt) != 0:
                    ret_transforms = cropper(mri1, 
                                             mri2, 
                                             gt,
                                             gr, 
                                             roi_size=(config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']),
                                             num_samples=config.hyperparameters['sample_per_mri'] if self.train else 1);
                else:
                    ret_transforms = self.crop_rand({'image1': mri1, 'image2': mri2,'mask': gt, 'gradient':gr, 'lbl' :np.ones_like(mri1)});
            
            ret_mri1 = None;
            ret_mri2 = None;
            ret_gt = None;
            ret_dt = None;

            for i in range(config.hyperparameters['sample_per_mri']):
                if config.hyperparameters['deterministic'] is False:
                    mri1_c = ret_transforms[i]['image1'];
                    mri2_c = ret_transforms[i]['image2'];
                    gr_c = ret_transforms[i]['gradient'];

                    gt_c = ret_transforms[i]['mask'];

                else:
                    center1 = [int(mri1.shape[1]//2)-32,int(mri1.shape[2]//2)-32, int(mri1.shape[3]//2)-32]
                    mri1_c = torch.from_numpy(mri1[:, int(center1[0]-32):int(center1[0]+32), int(center1[1]-32):int(center1[1]+32), int(center1[2]-32):int(center1[2]+32)]);
                    mri2_c = torch.from_numpy(mri2[:, int(center1[0]-32):int(center1[0]+32), int(center1[1]-32):int(center1[1]+32), int(center1[2]-32):int(center1[2]+32)]);
                    gt_c = torch.from_numpy(gt[:, int(center1[0]-32):int(center1[0]+32), int(center1[1]-32):int(center1[1]+32), int(center1[2]-32):int(center1[2]+32)]);
                    gr_c = torch.from_numpy(gr[:, int(center1[0]-32):int(center1[0]+32), int(center1[1]-32):int(center1[1]+32), int(center1[2]-32):int(center1[2]+32)]);

                total_heatmap = torch.zeros_like(mri2_c, dtype=torch.float64);

                gr_c = (np.abs(sobel(gaussian_filter(mri2_c, 1))));
                gr_c = torch.from_numpy(gr_c < threshold_otsu(gr_c))
                t1 =  torch.where(mri2_c>t, 0, 1);
                t2 = (mri2_c > threshold_otsu(mri2_c.numpy()));
                g = t1 * t2 * (gr_c);
                g = g.numpy();

                g = binary_opening(g.squeeze(), structure=np.ones((3,3,3))).astype(g.dtype)
                g = torch.from_numpy(np.expand_dims(g, axis=0));

                
                num_corrupted_patches = np.random.randint(1,5) if config.hyperparameters['deterministic'] is False else 3;
                for _ in range(num_corrupted_patches):
                    mri2_c, heatmap = add_synthetic_lesion_wm(mri2_c, g)
                    total_heatmap = torch.clamp(heatmap+total_heatmap, 0, 1);

                total_heatmap_thresh = torch.where(total_heatmap > 0.5, 1.0, 0.0);
                total_heatmap_thresh = torch.clamp(total_heatmap_thresh + gt_c, 0, 1);

                pos_dt = distance_transform_edt(np.where(total_heatmap_thresh.squeeze().numpy()==1, 0, 1));
                pos_dt = pos_dt/(np.max(pos_dt)+1e-4);

                neg_dt = distance_transform_edt(total_heatmap_thresh.squeeze().numpy()==1);
                neg_dt = neg_dt/(np.max(neg_dt)+1e-4);

                dt = pos_dt - neg_dt ;
                dt = torch.from_numpy(np.expand_dims(dt, axis = 0));
                
                if config.DEBUG_TRAIN_DATA:
                    pos_cords = np.where(total_heatmap_thresh >0.0);
                    if len(pos_cords[0]) != 0:
                        r = np.random.randint(0,len(pos_cords[0]));
                        center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
                    else:
                        center=[mri2_c.shape[1]//2, mri2_c.shape[2]//2, mri2_c.shape[3]//2]
                    visualize_2d([mri1_c, mri2_c, total_heatmap_thresh], center);
                
                mri1_c = self.transforms(mri1_c);

                #mri2_c = self.augment_noisy_image(mri2_c);
                mri2_c = self.transforms(mri2_c);

                if ret_mri1 is None:
                    ret_mri1 = mri1_c.unsqueeze(dim=0);
                    ret_mri2 = mri2_c.unsqueeze(dim=0);
                    ret_gt = total_heatmap_thresh.unsqueeze(dim=0);
                    ret_dt = dt.unsqueeze(dim=0);
                else:
                    ret_mri1 = torch.concat([ret_mri1, mri1_c.unsqueeze(dim=0)], dim=0);
                    ret_mri2 = torch.concat([ret_mri2, mri2_c.unsqueeze(dim=0)], dim=0);
                    ret_gt = torch.concat([ret_gt, total_heatmap_thresh.unsqueeze(dim=0)], dim=0);
                    ret_dt = torch.concat([ret_dt, dt.unsqueeze(dim=0)], dim=0);

        
            return ret_mri1, ret_mri2, ret_gt, ret_dt;
       
        else:
            mri1, mri2, ret_gt, patient_id, brainmask, loc = self.data[index];

            mri1 = np.expand_dims(mri1, axis=0);
            mri2 = np.expand_dims(mri2, axis=0);
            ret_gt = np.expand_dims(ret_gt, axis=0);

            ret_mri1 = self.transforms(mri1);
            ret_mri2 = self.transforms(mri2);

            if config.DEBUG_TRAIN_DATA:
                pos_cords = np.where(ret_gt == 1);
                if len(pos_cords[0]) != 0:
                    r = np.random.randint(0,len(pos_cords[0]));
                    center = [pos_cords[0][r], pos_cords[1][r],pos_cords[2][r]]
                else:
                    center=[mri1.shape[1]//2, mri1.shape[2]//2, mri1.shape[3]//2]
                visualize_2d([ret_mri1, ret_mri2, ret_gt, brainmask], center);

            return ret_mri1, ret_mri2, ret_gt, brainmask, patient_id, loc;
    def update_prediction(self, pred, patient_id, loc):
        self.pred_data[patient_id][(loc[0].item())*self.step_w:(loc[0].item())*self.step_w + config.hyperparameters['crop_size_w'], 
                                (loc[1].item())*self.step_h:((loc[1].item()))*self.step_h + config.hyperparameters['crop_size_h'], 
                                (loc[2].item())*self.step_d:((loc[2].item()))*self.step_d + config.hyperparameters['crop_size_d']] += np.array(pred.squeeze().detach().cpu().numpy()).astype("int32");

    def calculate_metrics(self, simple = True):
        ret = [];
        for k in tqdm(self.pred_data.keys()):
            if simple is True:
                dice = calculate_metric_percase(self.pred_data[k].squeeze(), self.gt_data[k].squeeze(), simple=simple);
            else:
                dice,hd,f1 = calculate_metric_percase(self.pred_data[k].squeeze(), self.gt_data[k].squeeze(), simple=simple);
            if np.sum(self.gt_data[k].squeeze()) > 0:
                ret.append(dice if simple is True else [dice, hd, f1]);
        return np.mean(ret) if simple is True else np.mean(np.array(ret), axis =0);


def update_folds(num_test_data = 200):
    if os.path.exists('cache') is False:
        os.makedirs('cache');

    mri_files = glob(os.path.join('mri_data','*.nii.gz'));
    patient_mri = dict();
    for mri_path in mri_files:
        patient_name = mri_path[:mri_path.find('-')];
        if patient_name not in patient_mri:
            patient_mri[patient_name] = [mri_path];
        else:
            patient_mri[patient_name].append(mri_path);

    kfold = KFold(5, random_state=42, shuffle=True);
    patient_list = list(patient_mri.keys());
    f = 0;
    for train_idx, test_idx in kfold.split(patient_list):
        train_mri = [patient_mri[patient_list[t]] for t in train_idx];
        train_mri = [item for pn in train_mri for item in pn];
        test_mri = [patient_mri[patient_list[t]] for t in test_idx];
        test_mri = [item for pn in test_mri for item in pn];
        test_mri = cache_test_dataset(num_test_data, test_mri, f);
        pickle.dump([train_mri, test_mri], open(f'cache/{f}.fold', 'wb'));
        f+=1;

def update_folds_isbi():
    if os.path.exists('cache_isbi') is False:
        os.makedirs('cache_isbi');

    #patient_ids = np.array([name for name in os.listdir('isbi') if os.path.isdir(f'isbi/{name}')]);
    patient_ids = glob('isbi/*/');
    patient_ids = np.array([p.replace('\\', '/')[:len(p)-1] for p in patient_ids])

    kfold = KFold(5, random_state=42, shuffle=True);
    f = 0;
    for train_idx, test_idx in kfold.split(patient_ids):
        train_ids, test_ids = patient_ids[train_idx], patient_ids[test_idx];

        pickle.dump([train_ids, test_ids], open(f'cache_isbi/{f}.fold', 'wb'));
        f+=1;

def update_folds_miccai():
    if os.path.exists('cache_miccai') is False:
        os.makedirs('cache_miccai');

    patient_ids = glob('miccai-processed/*/');
    patient_ids = np.array([p.replace('\\', '/')[:len(p)-1] for p in patient_ids])

    labels = [];
    for p in patient_ids:
        gt = nib.load(os.path.join(p, f'ground_truth.nii.gz'));
        gt = gt.get_fdata();
        lbl = np.sum(gt) > 0;
        labels.append(lbl);


    kfold = StratifiedKFold(5, random_state=42, shuffle=True);
    f = 0;
    for train_idx, test_idx in kfold.split(patient_ids,labels):
        train_ids, test_ids = patient_ids[train_idx], patient_ids[test_idx];

        pickle.dump([train_ids, test_ids], open(f'cache_miccai/{f}.fold', 'wb'));
        f+=1;

def get_loader(fold):
    
    train_mri, test_mri = pickle.load(open(f'cache/{fold}.fold', 'rb'));

    mri_dataset_train = MRI_Dataset(train_mri);
    train_loader = DataLoader(mri_dataset_train, 1, True, num_workers=config.hyperparameters['num_workers'], pin_memory=True);
    test_mri = glob(os.path.join('cache',f'{fold}','*.tstd'));
    mri_dataset_test = MRI_Dataset(test_mri, train=False);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=config.hyperparameters['num_workers'], pin_memory=True);

    return train_loader, test_loader;   

def get_loader_pretrain_miccai(fold):
    
    train_mri, test_mri = pickle.load(open(f'cache_miccai/{fold}.fold', 'rb'));

    with open(os.path.join('cache_miccai', f'fold{fold}.txt'), 'r') as f:
        train_ids = f.readline().rstrip();
        train_ids = train_ids.split(',');
        test_ids = f.readline().rstrip();
        test_ids = test_ids.split(',');
    train_ids =  [os.path.join('miccai-processed', t) for t in train_ids];
    test_ids = [os.path.join('miccai-processed', t) for t in test_ids];
    
    mri_paths = [];
    for t in train_ids:
        mri_paths.append(os.path.join(t, 'flair_time01_on_middle_space.nii.gz'));
        mri_paths.append(os.path.join(t, 'flair_time02_on_middle_space.nii.gz'));

    mri_dataset_train = MICCAI_PRETRAIN_Dataset(mri_paths);
    train_loader = DataLoader(mri_dataset_train, 1, True, num_workers=config.hyperparameters['num_workers'], pin_memory=True);
    test_mri = glob(os.path.join('cache_miccai',f'{fold}','*.tstd'));
    mri_dataset_test = MICCAI_PRETRAIN_Dataset(test_mri, train=False);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=config.hyperparameters['num_workers'], pin_memory=True);

    return train_loader, test_loader; 

def get_loader_miccai(fold):
    
    with open(os.path.join('cache_miccai', f'fold{fold}.txt'), 'r') as f:
        train_ids = f.readline().rstrip();
        train_ids = train_ids.split(',');
        test_ids = f.readline().rstrip();
        test_ids = test_ids.split(',');
    train_ids =  [os.path.join('miccai-processed', t) for t in train_ids];
    test_ids = [os.path.join('miccai-processed', t) for t in test_ids];



    mri_dataset_train = MICCAI_Dataset(train_ids[:1], train=True);
    train_loader = DataLoader(mri_dataset_train, 1, True, num_workers=config.hyperparameters['num_workers'], pin_memory=True);
    mri_dataset_test = MICCAI_Dataset(test_ids, train=False);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=config.hyperparameters['num_workers'], pin_memory=True);

    return train_loader, test_loader, mri_dataset_test; 

def standardize(img):
    img = img - np.min(img);
    img = (img / np.max(img))*255;
    return img;

def standardize(img):
    img = img - np.min(img);
    img = (img / np.max(img))*255;
    return img;

def visualize_2d(images, slice,):
    # if size is not None:
    #     res = Resize(size);
    #     mri_ret = res(mri.unsqueeze(dim=0));
    #     mri_ret = mri_ret.numpy().squeeze();
    fig, ax = plt.subplots(len(images),3);
    for i,img in enumerate(images):
        img = img.squeeze();
        ax[i][0].imshow(img[slice[0], :,:], cmap='gray');
        ax[i][1].imshow(img[:,slice[1],:], cmap='gray');
        ax[i][2].imshow(img[:,:,slice[2]], cmap='gray');
    plt.show()

def inpaint_3d(img, mask_g, num_corrupted_patches):
    
    mri = img;

    _,h,w,d = mri.shape;

    cubes = [];
    for n in range(num_corrupted_patches):
        mask_cpy = deepcopy(mask_g);
        size_x = np.random.randint(5,15) if config.hyperparameters['deterministic'] is False else 15;
        size_y = np.random.randint(5,20) if config.hyperparameters['deterministic'] is False else 15;
        size_z = np.random.randint(5,20) if config.hyperparameters['deterministic'] is False else 15;
        mask_cpy[:,:,:,d-size_z:] = 0;
        mask_cpy[:,:,:,:size_z+1] = 0;
        mask_cpy[:,:,w-size_y:,:] = 0;
        mask_cpy[:,:,:size_y+1,:] = 0;
        mask_cpy[:,h-size_x:,:,:] = 0;
        mask_cpy[:,:size_x+1,:,:] = 0;
        pos_cords = np.where(mask_cpy==1);

        if config.hyperparameters['deterministic'] is False:
            if len(pos_cords[0]) != 0:
                r = np.random.randint(0,len(pos_cords[0]));
                center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
            else:
                center = [img.shape[0]//2, img.shape[1]//2, img.shape[2]//2]
        else:
            center = [pos_cords[1][50], pos_cords[2][50],pos_cords[3][50]]
        
        cubes.append([center, size_x, size_y, size_z]);
    
 
    #shape
    cube = np.zeros((1,h,w,d), dtype=np.uint8);
    for c in cubes:
        cube[:,max(c[0][0]-c[1],0):min(c[0][0]+c[1], h), \
             max(c[0][1]-c[2],0):min(c[0][1]+c[2],w), \
             max(c[0][2]-c[3],0):min(c[0][2]+c[3],d)] = 1;

    #cube = transform(cube);
    cube_thresh = (cube>0)

    cube_thresh = GaussianSmooth(4, approx='erf')(cube_thresh);
    cube_thresh = cube_thresh / (torch.max(cube_thresh) + 1e-4);
    #================

    noise = GaussianSmooth(15)(mri);
    mri_after = (1-cube_thresh)*mri + (cube_thresh*noise);
    noise = GaussianSmooth(7)(mask_g.float());
    
    mri_after = torch.clip(mri_after, 0, 1);
    #mri_after = (mri_after*255).astype("uint8")
    #visualize_2d(mri_after, cube_thresh, slice=center[0:]);
    return mri_after, cube_thresh, noise, center;

def add_synthetic_lesion_wm(img, mask_g):
    
    mri = img;

    _,h,w,d = mri.shape;

    mask_cpy = deepcopy(mask_g);
    size_x = np.random.randint(2,6) if config.hyperparameters['deterministic'] is False else 3;
    size_y = size_x - np.random.randint(0,size_x-1) if config.hyperparameters['deterministic'] is False else 3;
    size_z = size_x - np.random.randint(0,size_x-1) if config.hyperparameters['deterministic'] is False else 3;
    mask_cpy[:,:,:,d-size_z:] = 0;
    mask_cpy[:,:,:,:size_z+1] = 0;
    mask_cpy[:,:,w-size_y:,:] = 0;
    mask_cpy[:,:,:size_y+1,:] = 0;
    mask_cpy[:,h-size_x:,:,:] = 0;
    mask_cpy[:,:size_x+1,:,:] = 0;
    pos_cords = np.where(mask_cpy==1);

    if config.hyperparameters['deterministic'] is False:
        if len(pos_cords[0]) != 0:
            r = np.random.randint(0,len(pos_cords[0]));
            center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
        else:
            center = [img.shape[1]//2, img.shape[2]//2, img.shape[3]//2]
    else:
        if len(pos_cords[0]) != 0:
            center = [pos_cords[1][int(len(pos_cords[0])//2)], pos_cords[2][int(len(pos_cords[0])//2)],pos_cords[3][int(len(pos_cords[0])//2)]]
        else:
            center = [img.shape[1]//2, img.shape[2]//2, img.shape[3]//2]
    
 
    #shape
    cube = torch.zeros((1,h,w,d), dtype=torch.uint8);
    cube[:,max(center[0]-size_x,0):min(center[0]+size_x, h), max(center[1]-size_y,0):min(center[1]+size_y,w), max(center[2]-size_z,0):min(center[2]+size_z,d)] = 1;
    cube = cube * mask_g;

    #var = np.random.uniform(1.0,1.2);
    cube = GaussianSmooth(1.2, approx='erf')(cube);
    cube = cube / (torch.max(cube) + 1e-4);
    #================

    noise = (torch.ones((1,h,w,d), dtype=torch.uint8));
    final = (cube)*(noise);
    mri_after = (1-cube)*mri + final;
    
    
    #noise = GaussianSmooth(7)(mask_g.float());
    #mri_after = torch.clip(mri_after, 0, 1);
    #mri_after = (mri_after*255).astype("uint8")
    #visualize_2d(mri_after, cube_thresh, slice=center[0:]);
    return mri_after, cube;

def cache_mri_gradients():
    patient_ids = glob(os.path.join('miccai-processed/*/'));
    for p in tqdm(patient_ids):
        patient_path = os.path.join(p, 'flair_time01_on_middle_space.nii.gz');

        file_name = os.path.basename(patient_path);
        file_name = file_name[:file_name.find('.')];
        mrimage = nib.load(patient_path)
        mrimage = mrimage.get_fdata()
        mrimage = window_center_adjustment(mrimage);
        mrimage = mrimage / np.max(mrimage);
        g = sobel(mrimage);
        g = g < threshold_otsu(g);
        pickle.dump(g, open(os.path.join(p, f'flair_time01_on_middle_space.gradient'), 'wb'));

        patient_path = os.path.join(p, 'flair_time02_on_middle_space.nii.gz');

        file_name = os.path.basename(patient_path);
        file_name = file_name[:file_name.find('.')];
        mrimage = nib.load(patient_path)
        mrimage = mrimage.get_fdata()
        mrimage = window_center_adjustment(mrimage);
        mrimage = mrimage / np.max(mrimage);
        g = sobel(mrimage);
        g = g < threshold_otsu(g);
        pickle.dump(g, open(os.path.join(p, f'flair_time02_on_middle_space.gradient'), 'wb'));


def gradient(mri):
    kernel1 = np.concatenate([np.ones((1, 3,3)), np.zeros((1, 3,3)), np.ones((1, 3,3))*-1], axis=0);
    kernel2 = np.concatenate([np.ones((3,1,3)), np.zeros((3,1,3)), np.ones((3,1,3))*-1], axis=1);
    kernel3 = np.concatenate([np.ones((3,3,1)), np.zeros((3,3,1)), np.ones((3,3,1))*-1], axis=2);


    img1 = convolve(mri, kernel1)
    img2 = convolve(mri, kernel2)
    img3 = convolve(mri, kernel3)
    ret = np.sqrt(img1**2 + img2**2 + img3**2);
    return ret;

def predict_on_miccai(base_path, model):

    normalize_internsity = NormalizeIntensity(subtrahend=0.5, divisor=0.5);
    def preprocess(mrimage):
        mask = mrimage > threshold_otsu(mrimage);
        mask = np.expand_dims(mask, axis=0);
        mrimage = mrimage / (np.max(mrimage)+1e-4);
        return mrimage, mask; 

    counter = 0;
    with torch.no_grad():
        gt_path = os.path.join(base_path,"ground_truth.nii.gz");
        brainmask_path = os.path.join(base_path, "brain_mask.nii.gz");

        first_mri_path = os.path.join(base_path, 'flair_time01_on_middle_space.nii.gz');
        second_mri_path = os.path.join(base_path, 'flair_time02_on_middle_space.nii.gz');

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

        step_w, step_h, step_d = config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d'];
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

                    # s = torch.sum(pred);

                    # #gt_lbl = torch.sum(pred).item()>0;
                    # if gt_lbl == 1:
                    #     dice = DiceLoss()(pred, ret_gt);
                    #     total_dice.append(dice.item());
                    # if gt_lbl ==0 and s > 0:
                    #     print('a');
                    
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
               
   
    final_pred = torch.from_numpy(predicted_aggregated);
    gt_padded = torch.from_numpy(gt_padded)
    return final_pred.unsqueeze(0).unsqueeze(0), gt_padded.unsqueeze(0).unsqueeze(0);