import os
from torch.utils.data import DataLoader, Dataset
import pickle
from glob import glob
import numpy as np
from copy import deepcopy, copy
import matplotlib.pyplot as plt

import torch
import config
from sklearn.model_selection import train_test_split, KFold
import nibabel as nib
from skimage.filters import gaussian, sobel, threshold_otsu, try_all_threshold
from monai.transforms import Compose, Resize, SpatialPadd, ScaleIntensityRange, Rand3DElastic, Resize, RandGaussianSmooth, OneOf, RandGibbsNoise, RandGaussianNoise, GaussianSmooth, NormalizeIntensity, RandCropByPosNegLabeld, GibbsNoise, RandSpatialCropSamplesd
from scipy.ndimage import convolve
from tqdm import tqdm
import math
from patchify import patchify
import seaborn as sns

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

def cropper(mri1, mri2, gt, roi_size, num_samples):
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

        ret.append(d);
    return ret;

class ISBI_Dataset(Dataset):
    def __init__(self, patient_ids, train = True) -> None:
        super().__init__();
        m1 = 0.7;
        m2 = 0.8;
        self.augment_noisy_image = OneOf([
            RandGaussianSmooth(prob=1.0, sigma_x=(m1,m2), sigma_y=(m1,m2), sigma_z=(m1,m2)),
            RandGaussianNoise(prob=1.0,std=.1),
            RandGibbsNoise(prob=1.0, alpha=(0.65,0.75))
        ], weights=[1,1,1])


        self.transforms = Compose(
            [
                
                NormalizeIntensity(subtrahend=0.5, divisor=0.5)
            ]
        )
        
        self.crop = Compose(
            [ 
                SpatialPadd(keys=['image', 'mask'], spatial_size = [config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']]),
                RandCropByPosNegLabeld(
                keys=['image',  'mask'], 
                label_key='mask', 
                spatial_size= (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']),
                pos=1, 
                neg=0,
                num_samples=config.hyperparameters['sample_per_mri'] if train else 1)
            ]
        )

        self.train = train;

        self.mri = [];

        if train:
            for patient_path in patient_ids:
                patient_id = patient_path[patient_path.rfind('/')+1:]
                num_mri = len(os.listdir(os.path.join(patient_path, 'preprocessed')))//4;
                curr_mri = [];
                curr_gt = [];
                for n in range(1,int(num_mri)+1):
                    mri = nib.load(os.path.join(patient_path, 'preprocessed', f'{patient_id}_0{n}_flair_pp.nii'));
                    mri = mri.get_fdata();
                    mri = window_center_adjustment(mri);
                    gt = nib.load(os.path.join(patient_path, 'masks', f'{patient_id}_0{n}_mask1.nii'));
                    curr_mri.append(mri);
                    curr_gt.append(gt.get_fdata());
        
                self.mri.append([curr_mri, curr_gt]);

        else:
            mri_temp = [];
            gt_temp = [];
            for patient_path in patient_ids:
                patient_id = patient_path[patient_path.rfind('/')+1:]
                num_mri = len(os.listdir(os.path.join(patient_path, 'preprocessed')))//4;
                curr_mri = [];
                curr_gt = [];
                for n in range(1,int(num_mri)+1):
                    mri = nib.load(os.path.join(patient_path, 'preprocessed', f'{patient_id}_0{n}_flair_pp.nii'));
                    mri = mri.get_fdata();
                    mri = window_center_adjustment(mri);
                    gt = nib.load(os.path.join(patient_path, 'masks', f'{patient_id}_0{n}_mask2.nii'));
                    gt = gt.get_fdata();
                    

                    w,h,d = mri.shape;
                    new_w = math.ceil(w / config.hyperparameters['crop_size_w']) * config.hyperparameters['crop_size_w'];
                    new_h = math.ceil(h / config.hyperparameters['crop_size_h']) * config.hyperparameters['crop_size_h'];
                    new_d = math.ceil(d / config.hyperparameters['crop_size_d']) * config.hyperparameters['crop_size_d'];

                    mri_padded  = np.zeros((new_w, new_h, new_d), dtype = mri.dtype);
                    gt_padded  = np.zeros((new_w, new_h, new_d), dtype = gt.dtype);

                    mri_padded[:w,:h,:d] = mri;
                    gt_padded[:w,:h,:d] = gt;

                    step_w, step_h, step_d = config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d'];
                    mri_patches = patchify(mri_padded, 
                                                        (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                        (step_w, step_h, step_d));
                    gt_patches = patchify(gt_padded, 
                                                        (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']), 
                                                        (step_w, step_h, step_d));
                    mri_patches = mri_patches.reshape(mri_patches.shape[0]*mri_patches.shape[1]*mri_patches.shape[2], mri_patches.shape[3], mri_patches.shape[4],mri_patches.shape[5]);
                    gt_patches = gt_patches.reshape(gt_patches.shape[0]*gt_patches.shape[1]*gt_patches.shape[2], gt_patches.shape[3], gt_patches.shape[4],gt_patches.shape[5]);
                
                    curr_mri.append(mri_patches);
                    curr_gt.append(gt_patches);
                mri_temp.append(curr_mri);
                gt_temp.append(curr_gt);
            
            temp = [];
            for l in range(len(mri_temp)):
                for i in range(len(mri_temp[l])):
                    for j in range(i+1,len(mri_temp[l])):
                        mri1 = mri_temp[l][i];
                        mri2 = mri_temp[l][j];
                        gt1 = gt_temp[l][i];
                        gt2 = gt_temp[l][j];
                        t = [[[mri1[k],mri2[k], gt1[k], gt2[k]], np.clip(gt1[k] + gt2[k], 0, 1) - (gt1[k])*(gt2[k])] for k in range(mri1.shape[0])]
                        temp.extend(t);
            self.mri = temp;
    
    def __len__(self):
        return len(self.mri);

    def __getitem__(self, index):
        if self.train:
            
            mri, gt = self.mri[index];

            mri = mri[0];
            gt = gt[0];
            mri = np.expand_dims(mri, axis=0);
            mri = mri / (np.max(mri)+1e-4);
            gt = np.expand_dims(gt, axis=0);

            ret_mri1 = None;
            ret_mri2 = None;
            ret_gt = None;
            
            if config.hyperparameters['deterministic'] is False:
                ret_transforms = self.crop({'image': mri, 'mask': gt});
            
            for i in range(config.hyperparameters['sample_per_mri']):
                if config.hyperparameters['deterministic'] is False:
                    mri_c = ret_transforms[i]['image'];
                    gt_c = ret_transforms[i]['mask'];
                    mrimage_noisy = copy(mri_c);
                else:
                    center = [68, 139, 83]
                    mri_c = mri[:, int(center[0]-64):int(center[0]+64), int(center[1]-64):int(center[1]+64), int(center[2]-32):int(center[2]+32)];
                    gt_c = gt[:, int(center[0]-64):int(center[0]+64), int(center[1]-64):int(center[1]+64), int(center[2]-32):int(center[2]+32)];
                    mri_c = torch.from_numpy(mri_c);
                    mrimage_noisy = copy(mri_c);

                total_heatmap = torch.zeros_like(mrimage_noisy, dtype=torch.float64);


                num_corrupted_patches = np.random.randint(1,5) if config.hyperparameters['deterministic'] is False else 3;
                for i in range(num_corrupted_patches):
                    mrimage_noisy, heatmap, noise, center = add_synthetic_lesion_wm(mrimage_noisy, gt_c)
                    total_heatmap += heatmap;
                # total_heatmap_thresh = total_heatmap > 0;
                total_heatmap_thresh = torch.where(total_heatmap > 0.5, 1.0, 0.0);

                pos_cords = np.where(total_heatmap_thresh == 1);
                r = np.random.randint(0,len(pos_cords[0]));
                center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]

                if config.hyperparameters['deterministic'] is True:
                    mrimage_noisy = GibbsNoise(alpha = 0.65)(mrimage_noisy);
                else:
                    mrimage_noisy = self.augment_noisy_image(mrimage_noisy);
                
                #visualize_2d([mri_c, mrimage_noisy, total_heatmap_thresh, noise], center);

                mri_c = self.transforms(mri_c);
                mrimage_noisy = self.transforms(mrimage_noisy);
                
                

                if ret_mri1 is None:
                    ret_mri1 = mri_c.unsqueeze(dim=0);
                    ret_mri2 = mrimage_noisy.unsqueeze(dim=0);
                    ret_gt = total_heatmap_thresh.unsqueeze(dim=0);
                else:
                    ret_mri1 = torch.concat([ret_mri1, mri_c.unsqueeze(dim=0)], dim=0);
                    ret_mri2 = torch.concat([ret_mri2, mrimage_noisy.unsqueeze(dim=0)], dim=0);
                    ret_gt = torch.concat([ret_gt, total_heatmap_thresh.unsqueeze(dim=0)], dim=0);
            
            return ret_mri1, ret_mri2, ret_gt;
        else:
            mri, ret_gt = self.mri[index];
            ret_mri1, ret_mri2, gt1, gt2 = mri[0], mri[1], mri[2], mri[3];

            ret_mri1 = ret_mri1 / (np.max(ret_mri1)+1e-4);
            ret_mri2 = ret_mri2 / (np.max(ret_mri2)+1e-4);
            ret_mri1 +gt1;
            ret_mri2 +gt2;

            ret_mri1 = self.transforms(ret_mri1);
            ret_mri2 = self.transforms(ret_mri2);


            # pos_cords = np.where(ret_gt == 1);
            # if len(pos_cords[0]) != 0:
            #     r = np.random.randint(0,len(pos_cords[0]));
            #     center = [pos_cords[0][r], pos_cords[1][r],pos_cords[2][r]]
            # else:
            #     center=[10,10,10]
            # visualize_2d([ret_mri1, ret_mri2,ret_mri1 +gt1, ret_mri2 +gt2, ret_gt], center);

            return ret_mri1, ret_mri2, ret_gt;

class MICCAI_Dataset(Dataset):
    def __init__(self, patient_ids, train = True) -> None:
        super().__init__();
        m1 = 0.7;
        m2 = 0.8;
        self.augment_noisy_image = OneOf([
            RandGaussianSmooth(prob=1.0, sigma_x=(m1,m2), sigma_y=(m1,m2), sigma_z=(m1,m2)),
            RandGaussianNoise(prob=1.0,std=.1),
            RandGibbsNoise(prob=1.0, alpha=(0.65,0.75))
        ], weights=[1,1,1])


        self.transforms = Compose(
            [
                
                NormalizeIntensity(subtrahend=0.5, divisor=0.5)
            ]
        )
        
        self.crop_pos_neg = Compose(
            [ 
                SpatialPadd(keys=['image1', 'image2','mask'], spatial_size = [config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']]),
                RandCropByPosNegLabeld(
                keys=['image1', 'image2','mask'], 
                label_key='mask', 
                spatial_size= (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']),
                pos=1, 
                neg=0,
                num_samples=config.hyperparameters['sample_per_mri'] if train else 1)
            ]
        )

        self.crop_rand = Compose(
            [ 
                SpatialPadd(keys=['image1', 'image2','mask'], spatial_size = [config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']]),
                RandCropByPosNegLabeld(
                keys=['image1', 'image2', 'mask', 'lbl'], 
                label_key='lbl', 
                spatial_size= (config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']),
                pos=1, 
                neg=0,
                num_samples=config.hyperparameters['sample_per_mri'] if train else 1)
            ]
        )

        self.train = train;

        self.data = [];
        
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

                gt = gt * brainmask;
                
                self.data.append([mri1, mri2, gt, patient_path]);
        
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
                mri1_patches = mri1_patches.reshape(mri1_patches.shape[0]*mri1_patches.shape[1]*mri1_patches.shape[2], mri1_patches.shape[3], mri1_patches.shape[4],mri1_patches.shape[5]);
                mri2_patches = mri2_patches.reshape(mri2_patches.shape[0]*mri2_patches.shape[1]*mri2_patches.shape[2], mri2_patches.shape[3], mri2_patches.shape[4],mri2_patches.shape[5]);
                gt_patches = gt_patches.reshape(gt_patches.shape[0]*gt_patches.shape[1]*gt_patches.shape[2], gt_patches.shape[3], gt_patches.shape[4],gt_patches.shape[5]);
                brainmask_patches = brainmask_patches.reshape(brainmask_patches.shape[0]*brainmask_patches.shape[1]*brainmask_patches.shape[2], brainmask_patches.shape[3], brainmask_patches.shape[4],brainmask_patches.shape[5]);

                curr_data = [];
                for i in range(mri1_patches.shape[0]):
                    if np.sum(mri1_patches[i]) != 0 and np.sum(mri2_patches[i]) != 0:
                        curr_data.append((mri1_patches[i],mri2_patches[i],gt_patches[i], 0 if np.sum(gt_patches[i]) == 0 else 1, brainmask_patches[i]))

                self.data.extend(curr_data);

            
    
    def __len__(self):
        return len(self.data);

    def __getitem__(self, index):
        if self.train:
            
            mri1, mri2, gt, pp = self.data[index];

            mri1 = np.expand_dims(mri1, axis=0);
            mri2 = np.expand_dims(mri2, axis=0);
            gt = np.expand_dims(gt, axis=0);

            mri1 = mri1 / (np.max(mri1)+1e-4);
            mri2 = mri2 / (np.max(mri2)+1e-4);

            if config.hyperparameters['deterministic'] is False:
                if np.sum(gt) != 0:

                    ret_transforms = cropper(mri1, 
                                             mri2, 
                                             gt, 
                                             roi_size=(config.hyperparameters['crop_size_w'], config.hyperparameters['crop_size_h'], config.hyperparameters['crop_size_d']),
                                             num_samples=config.hyperparameters['sample_per_mri'] if self.train else 1);
                else:

                    ret_transforms = self.crop_rand({'image1': mri1, 'image2': mri2,'mask': gt, 'lbl' :np.ones_like(mri1)});
            
            ret_mri1 = None;
            ret_mri2 = None;
            ret_gt = None;

            for i in range(config.hyperparameters['sample_per_mri']):
                if config.hyperparameters['deterministic'] is False:
                    mri1_c = ret_transforms[i]['image1'];
                    mri2_c = ret_transforms[i]['image2'];
                    gt_c = ret_transforms[i]['mask'];
                
                    
                    #mrimage_noisy = copy(mri_c);
                else:
                    center = [int(mri1.shape[1]//2),int(mri1.shape[2]//2), int(mri1.shape[3]//2)]
                    mri1_c = torch.from_numpy(mri1[:, int(center[0]-64):int(center[0]+64), int(center[1]-64):int(center[1]+64), int(center[2]-32):int(center[2]+32)]);
                    mri2_c = torch.from_numpy(mri2[:, int(center[0]-64):int(center[0]+64), int(center[1]-64):int(center[1]+64), int(center[2]-32):int(center[2]+32)]);
                    gt_c = torch.from_numpy(gt[:, int(center[0]-64):int(center[0]+64), int(center[1]-64):int(center[1]+64), int(center[2]-32):int(center[2]+32)]);

                pos_cords = np.where(gt_c > 0);
                if len(pos_cords[0]) != 0:
                    r = np.random.randint(0,len(pos_cords[0]));
                    center = [pos_cords[0][r], pos_cords[1][r],pos_cords[2][r]]
                else:
                    center = [mri1_c.shape[0]//2, mri1_c.shape[1]//2, mri1_c.shape[2]//2]

                total_heatmap = torch.zeros_like(mri2_c, dtype=torch.float64);
                g = (mri2_c > threshold_otsu(mri2_c.numpy())) *  torch.where(mri2_c<0.8, 1.0, 0.0);
                num_corrupted_patches = np.random.randint(1,5) if config.hyperparameters['deterministic'] is False else 3;
                for i in range(num_corrupted_patches):
                    mri2_c, heatmap, noise, center = add_synthetic_lesion_wm(mri2_c, g)
                    total_heatmap += heatmap;
                
                mri1_c = self.transforms(mri1_c);

                mri2_c = self.augment_noisy_image(mri2_c);
                mri2_c = self.transforms(mri2_c);

                total_heatmap_thresh = torch.where(total_heatmap > 0.5, 1.0, 0.0);
                total_heatmap_thresh += gt_c;

                # pos_cords = np.where(total_heatmap_thresh == 1);
                # r = np.random.randint(0,len(pos_cords[0]));
                # center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]

                # visualize_2d([mri1_c, mri2_c, total_heatmap_thresh, g], center);
                if ret_mri1 is None:
                    ret_mri1 = mri1_c.unsqueeze(dim=0);
                    ret_mri2 = mri2_c.unsqueeze(dim=0);
                    ret_gt = total_heatmap_thresh.unsqueeze(dim=0);
                else:
                    ret_mri1 = torch.concat([ret_mri1, mri1_c.unsqueeze(dim=0)], dim=0);
                    ret_mri2 = torch.concat([ret_mri2, mri2_c.unsqueeze(dim=0)], dim=0);
                    ret_gt = torch.concat([ret_gt, total_heatmap_thresh.unsqueeze(dim=0)], dim=0);
        
            return ret_mri1, ret_mri2, ret_gt;
       
        else:
            mri1, mri2, ret_gt, lbl, brainmask = self.data[index];

            mri1 = np.expand_dims(mri1, axis=0);
            mri2 = np.expand_dims(mri2, axis=0);
            ret_gt = np.expand_dims(ret_gt, axis=0);

            ret_mri1 = self.transforms(mri1);
            ret_mri2 = self.transforms(mri2);


            # pos_cords = np.where(ret_gt == 1);
            # if len(pos_cords[0]) != 0:
            #     r = np.random.randint(0,len(pos_cords[0]));
            #     center = [pos_cords[0][r], pos_cords[1][r],pos_cords[2][r]]
            # else:
            #     center=[mri1.shape[1]//2, mri1.shape[2]//2, mri1.shape[3]//2]
            # visualize_2d([ret_mri1, ret_mri2, ret_gt, brainmask], center);

            return ret_mri1, ret_mri2, ret_gt, lbl, brainmask;

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

    kfold = KFold(5, random_state=42, shuffle=True);
    f = 0;
    for train_idx, test_idx in kfold.split(patient_ids):
        train_ids, test_ids = patient_ids[train_idx], patient_ids[test_idx];

        pickle.dump([train_ids, test_ids], open(f'cache_miccai/{f}.fold', 'wb'));
        f+=1;

def get_loader(fold):
    
    train_mri, test_mri = pickle.load(open(f'cache/{fold}.fold', 'rb'));

    mri_dataset_train = MRI_Dataset(train_mri);
    train_loader = DataLoader(mri_dataset_train, 1, True, num_workers=0, pin_memory=True);
    test_mri = glob(os.path.join('cache',f'{fold}','*.tstd'));
    mri_dataset_test = MRI_Dataset(test_mri, train=False);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=0, pin_memory=True);

    return train_loader, test_loader;   

def get_loader_isbi(fold):
    
    train_ids, test_ids = pickle.load(open(os.path.join(f'cache_isbi',f'{fold}.fold'), 'rb'));

    mri_dataset_train = ISBI_Dataset(train_ids, train=False);
    train_loader = DataLoader(mri_dataset_train, config.hyperparameters['batch_size'], True, num_workers=8, pin_memory=True);
    mri_dataset_test = ISBI_Dataset(test_ids, train=False);
    test_loader = DataLoader(mri_dataset_test, 1, True, num_workers=8, pin_memory=True);

    return train_loader, test_loader;   

def get_loader_miccai(fold):
    
    train_ids, test_ids = pickle.load(open(os.path.join(f'cache_miccai',f'{fold}.fold'), 'rb'));

    mri_dataset_train = MICCAI_Dataset(train_ids, train=True);
    train_loader = DataLoader(mri_dataset_train, 1, True, num_workers=8, pin_memory=True);
    mri_dataset_test = MICCAI_Dataset(test_ids, train=False);
    test_loader = DataLoader(mri_dataset_test, 1, True, num_workers=8, pin_memory=True);

    return train_loader, test_loader; 

def standardize(img):
    img = img - np.min(img);
    img = (img / np.max(img))*255;
    return img;

def visualize_2d(images, slice, size=None,):
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
    plt.show();

def inpaint_3d(img, mask_g):
    
    mri = img;

    _,h,w,d = mri.shape;

    mask_cpy = deepcopy(mask_g);
    size_x = np.random.randint(10,25) if config.hyperparameters['deterministic'] is False else 15;
    size_y = np.random.randint(10,35) if config.hyperparameters['deterministic'] is False else 15;
    size_z = np.random.randint(10,35) if config.hyperparameters['deterministic'] is False else 15;
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
    
 
    #shape
    cube = np.zeros((1,h,w,d), dtype=np.uint8);
    cube[:,max(center[0]-size_x,0):min(center[0]+size_x, h), max(center[1]-size_y,0):min(center[1]+size_y,w), max(center[2]-size_z,0):min(center[2]+size_z,d)] = 1;

    #cube = transform(cube);
    cube_thresh = (cube>0)

    cube_thresh = GaussianSmooth(7, approx='erf')(cube_thresh);
    cube_thresh = cube_thresh / (torch.max(cube_thresh) + 1e-4);
    #================

    noise = GaussianSmooth(7)(mri);
    final = (cube_thresh)*(noise);
    noise = GaussianSmooth(7)(mask_g.float());
    mri_after = (1-cube_thresh)*mri + final;
    
    mri_after = torch.clip(mri_after, 0, 1);
    #mri_after = (mri_after*255).astype("uint8")
    #visualize_2d(mri_after, cube_thresh, slice=center[0:]);
    return mri_after, cube_thresh, noise, center;

def add_synthetic_lesion_wm(img, mask_g):
    
    mri = img;

    _,h,w,d = mri.shape;

    mask_cpy = deepcopy(mask_g);
    size_x = np.random.randint(1,5) if config.hyperparameters['deterministic'] is False else 3;
    size_y = np.random.randint(1,5) if config.hyperparameters['deterministic'] is False else 3;
    size_z = np.random.randint(1,5) if config.hyperparameters['deterministic'] is False else 3;
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
        center = [pos_cords[1][50], pos_cords[2][50],pos_cords[3][50]]
    
 
    #shape
    cube = torch.zeros((1,h,w,d), dtype=torch.uint8);
    cube[:,max(center[0]-size_x,0):min(center[0]+size_x, h), max(center[1]-size_y,0):min(center[1]+size_y,w), max(center[2]-size_z,0):min(center[2]+size_z,d)] = 1;
    cube = cube * mask_g;


    cube = GaussianSmooth(2, approx='erf')(cube);
    cube = cube / (torch.max(cube) + 1e-4);
    #================

    noise = (torch.ones((1,h,w,d), dtype=torch.uint8));
    final = (cube)*(noise);
    mri_after = (1-cube)*mri + final;
    
    
    #noise = GaussianSmooth(7)(mask_g.float());
    #mri_after = torch.clip(mri_after, 0, 1);
    #mri_after = (mri_after*255).astype("uint8")
    #visualize_2d(mri_after, cube_thresh, slice=center[0:]);
    return mri_after, cube, noise, center;

def cache_mri_gradients():
    all_mri = glob(os.path.join('mri_data', '*.nii.gz'));
    for mri_file in tqdm(all_mri):
        file_name = os.path.basename(mri_file);
        file_name = file_name[:file_name.find('.')];
        mrimage = nib.load(mri_file)
        mrimage = nib.as_closest_canonical(mrimage);
        mrimage = mrimage.get_fdata()
        mrimage = window_center_adjustment(mrimage);
        mrimage = mrimage / np.max(mrimage);
        g = gradient(mrimage);
        g = g > threshold_otsu(g);
        pickle.dump(g, open(os.path.join('mri_data', f'{file_name}.gradient'), 'wb'));


def gradient(mri):
    kernel1 = np.concatenate([np.ones((1, 3,3)), np.zeros((1, 3,3)), np.ones((1, 3,3))*-1], axis=0);
    kernel2 = np.concatenate([np.ones((3,1,3)), np.zeros((3,1,3)), np.ones((3,1,3))*-1], axis=1);
    kernel3 = np.concatenate([np.ones((3,3,1)), np.zeros((3,3,1)), np.ones((3,3,1))*-1], axis=2);


    img1 = convolve(mri, kernel1)
    img2 = convolve(mri, kernel2)
    img3 = convolve(mri, kernel3)
    ret = np.sqrt(img1**2 + img2**2 + img3**2);
    return ret;