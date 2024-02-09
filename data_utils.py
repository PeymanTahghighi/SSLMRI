import os
from torch.utils.data import DataLoader, Dataset
import pickle
from glob import glob
import numpy as np
from copy import deepcopy, copy
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import nibabel as nib
from skimage.filters import sobel, threshold_otsu
from monai.transforms import Compose, Resize, Resize, RandGaussianSmooth, OneOf, RandGibbsNoise, RandGaussianNoise, GaussianSmooth, NormalizeIntensity, RandCropByPosNegLabeld, GibbsNoise, RandSpatialCropSamplesd
from scipy.ndimage import binary_opening
from tqdm import tqdm
import math
from patchify import patchify
from scipy.ndimage import distance_transform_edt, sobel, gaussian_filter
from utility import calculate_metric_percase

def window_center_adjustment(img):
    """window center adjustment, similar to what ITKSnap does

    Parameters
    ----------
    img : np.ndarray
        input image

    """
    hist = np.histogram(img.ravel(), bins = int(np.max(img)))[0];
    hist = hist / (hist.sum()+1e-4);
    hist = np.cumsum(hist);

    hist_thresh = ((1-hist) < 5e-4);
    max_intensity = np.where(hist_thresh == True)[0][0];
    adjusted_img = img * (255/(max_intensity + 1e-4));
    adjusted_img = np.where(adjusted_img > 255, 255, adjusted_img).astype("uint8");

    return adjusted_img;

def cache_dataset_miccai(args):
    """cache testing dataset for self-supervised pretraining model

    Parameters
    ----------
    args : dict
        arguments

    """
    if os.path.exists(f'cache_miccai-2016') is False:
        os.makedirs(f'cache_miccai-2016');
    
    training_centers = ['01', '07', '08'];
    testing_centers = ['01', '03', '07', '08'];
    all_mri_path = [];
    for tc in training_centers:
        patients = glob(os.path.join('miccai-2016', 'Training', f'Center_{tc}','*/'));
        for p in patients:
            all_mri_path.append(os.path.join(p, 'Preprocessed_Data', 'FLAIR_preprocessed.nii.gz'));
    
    for tc in testing_centers:
        patients = glob(os.path.join('miccai-2016', 'Testing', f'Center_{tc}','*/'));
        for p in patients:
            all_mri_path.append(os.path.join(p, 'Preprocessed_Data', 'FLAIR_preprocessed.nii.gz'));
    
    train_ids, test_ids = train_test_split(all_mri_path, test_size=0.1, shuffle=True, random_state=42);

    
    mri_dataset_test = MICCAI_PRETRAIN_Dataset(args, test_ids,cache=True);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=0, pin_memory=True);

    test_ids = [];
    num_data = math.ceil(args.num_cache_data/len(test_loader));
    counter = 0;
    ret = [];
    for n in tqdm(range(num_data)):
        for (batch) in test_loader:
            ret.append(os.path.join('cache_miccai-2016', f'{counter}.tstd'))
            pickle.dump([b.squeeze() for b in batch], open(os.path.join('cache_miccai-2016', f'{counter}.tstd'), 'wb'));
            test_ids.append(os.path.join('cache_miccai-2016', f'{counter}.tstd'));
            counter += 1;

    pickle.dump([train_ids, test_ids], open(os.path.join('cache_miccai-2016', 'train_test_split.dmp'), 'wb'));
    return ret;

def cropper(mri1, 
            mri2, 
            gt, 
            gr, 
            roi_size,
            num_samples):
    """crop two time-points MRI scans at the same time for new lesion segmentation model

    Parameters
    ----------
    mri1 : np.ndarray
        first MRI scan

    mri2 : np.ndarray
        second MRI scan

    gt : np.ndarray
        ground truth segmentations

    gr : np.ndarray
        gradients of MRI scan

    rot_size : list
        size to crop for three axis

    num_samples : int
        number of samples to crop

    """
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
    """Dataset for self-supervised pretraining

    it returns examples for training which includes to MRI patch and one ground truth labels

    Parameters
    ----------
    args : dict
        arguments

    mr_images : list
        list of mri images, should a string list

    train : bool
        indicate if we are in training or testing mode

    cache: bool
        inidicate if we are only caching dataset or we are using it for training

    Attributes
    ----------
    mr_imges : list
        list of loaded mri scans

    """
    def __init__(self, 
                 args, 
                 mr_images, 
                 train = True,
                 cache = False) -> None:
        super().__init__();

        self.args = args;
        m1 = 0.7;
        m2 = 0.8;
        self.augment_noisy_image = OneOf([
            RandGaussianSmooth(prob=1.0, sigma_x=(m1,m2), sigma_y=(m1,m2), sigma_z=(m1,m2)),
            RandGaussianNoise(prob=1.0,std=0.05),
            RandGibbsNoise(prob=1.0, alpha=(0.65,0.75))
        ], weights=[1,1,1])


        self.transforms = NormalizeIntensity(subtrahend=0.5, divisor=0.5);
        self.crop = RandCropByPosNegLabeld(
            keys=['image', 'gradient', 'thresh', 'mask'], 
            label_key='mask', 
            spatial_size= (args.crop_size_w, args.crop_size_h, args.crop_size_d),
            pos=1, 
            neg=0,
            num_samples=1 if cache else args.sample_per_mri if train else 1);
        self.resize = Resize(spatial_size=[args.crop_size_w, args.crop_size_h, args.crop_size_d]);

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
            mrimage, gr = self.mr_images[index];
            mask = mrimage > threshold_otsu(mrimage);            
            mask = np.expand_dims(mask, axis=0);
            mrimage = np.expand_dims(mrimage, axis=0);
            mrimage = mrimage / (np.max(mrimage)+1e-4);

            g = (mrimage > 0.9);
            g = torch.from_numpy(g);
            ret_mrimage = None;
            ret_mrimage_noisy = None;
            ret_total_heatmap = None;

            if self.args.deterministic is False:
                ret_transforms = self.crop({'image': mrimage,'gradient': g, 'thresh': mask, 'mask': g});
            
            for i in range(self.args.sample_per_mri if self.cache is False else 1):
                if self.args.deterministic is False:
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
                    
                    mask_c = torch.from_numpy(mask_c);

                total_heatmap = torch.zeros_like(mrimage_noisy, dtype=torch.float64);

                num_corrupted_patches = np.random.randint(1,5) if self.args.deterministic is False else 3;

                mrimage_noisy, heatmap, noise, center = inpaint_3d(mrimage_noisy, g_c, num_corrupted_patches, self.args.deterministic)
                total_heatmap += heatmap;
                
                total_heatmap = total_heatmap * mask_c;
                mrimage_noisy = mrimage_noisy * mask_c;
                mrimage_c = mrimage_c * mask_c;

                total_heatmap_thresh = torch.where(total_heatmap > 0.5, 1.0, 0.0);
                part_first = mrimage_c * total_heatmap_thresh;
                part_second = mrimage_noisy * total_heatmap_thresh;
                if self.args.deterministic is True:
                    mrimage_noisy = GibbsNoise(alpha = 0.65)(mrimage_noisy);
                else:
                    mrimage_noisy = self.augment_noisy_image(mrimage_noisy);

                diff = torch.abs(part_first - part_second) > (0.2);

                total_heatmap_thresh = torch.where(diff > 0, 0, 1);
                
                if self.args.debug_train_data:
                    visualize_2d([mrimage_c, mrimage_noisy, total_heatmap, diff, g_c], center);
                
                mrimage_c = self.transforms(mrimage_c)[0];
                mrimage_noisy = self.transforms(mrimage_noisy)[0];

                if ret_mrimage is None:
                    ret_mrimage = mrimage_c.unsqueeze(dim=0);
                    ret_mrimage_noisy = mrimage_noisy.unsqueeze(dim=0);

                    ret_total_heatmap = total_heatmap_thresh.unsqueeze(dim=0);
                else:
                    ret_mrimage = torch.concat([ret_mrimage, mrimage_c.unsqueeze(dim=0)], dim=0);
                    ret_mrimage_noisy = torch.concat([ret_mrimage_noisy, mrimage_noisy.unsqueeze(dim=0)], dim=0);

                    ret_total_heatmap = torch.concat([ret_total_heatmap, total_heatmap_thresh.unsqueeze(dim=0)], dim=0);
            
            return ret_mrimage, ret_mrimage_noisy, ret_total_heatmap;
        else:
            ret = self.mr_images[index];
            return ret;

class MICCAI_Dataset(Dataset):
    """Dataset for self-supervised pretraining

    it returns examples for training which includes to MRI patch and one ground truth labels

    Parameters
    ----------
    args : dict
        arguments

    patient_ids : list
        list of mri images, should a string list

    train : bool
        indicate if we are in training or testing mode

    Attributes
    ----------
    mr_imges : list
        list of loaded mri scans

    """
    def __init__(self, 
                 args, 
                 patient_ids, 
                 train = True) -> None:
        super().__init__();
        self.args = args;
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
                spatial_size= (args.crop_size_w, args.crop_size_h, args.crop_size_d),
                pos=1, 
                neg=0,
                num_samples=args.sample_per_mri if train else 1,)
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

                w,h,d = mri1.shape;
                new_w = math.ceil(w / args.crop_size_w) * args.crop_size_w;
                new_h = math.ceil(h / args.crop_size_h) * args.crop_size_h;
                new_d = math.ceil(d / args.crop_size_d) * args.crop_size_d;

                mri1_padded  = np.zeros((new_w, new_h, new_d), dtype = mri1.dtype);
                mri2_padded  = np.zeros((new_w, new_h, new_d), dtype = mri2.dtype);
                gt_padded  = np.zeros((new_w, new_h, new_d), dtype = gt.dtype);
                brainmask_padded  = np.zeros((new_w, new_h, new_d), dtype = brainmask.dtype);

                mri1_padded[:w,:h,:d] = mri1;
                mri2_padded[:w,:h,:d] = mri2;
                gt_padded[:w,:h,:d] = gt;
                brainmask_padded[:w,:h,:d] = brainmask;

                self.step_w, self.step_h, self.step_d = args.crop_size_w, args.crop_size_h, args.crop_size_d;
                mri1_patches = patchify(mri1_padded, 
                                                    (args.crop_size_w, args.crop_size_h, args.crop_size_d), 
                                                    (self.step_w, self.step_h, self.step_d));
                mri2_patches = patchify(mri2_padded, 
                                                    (args.crop_size_w, args.crop_size_h, args.crop_size_d), 
                                                    (self.step_w, self.step_h, self.step_d));
                gt_patches = patchify(gt_padded, 
                                                    (args.crop_size_w, args.crop_size_h, args.crop_size_d), 
                                                    (self.step_w, self.step_h, self.step_d));
                brainmask_patches = patchify(brainmask_padded, 
                                                    (args.crop_size_w, args.crop_size_h, args.crop_size_d), 
                                                    (self.step_w, self.step_h, self.step_d));
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
            

            if self.args.deterministic is False:
                if np.sum(gt) != 0:
                    ret_transforms = cropper(mri1, 
                                             mri2, 
                                             gt,
                                             gr, 
                                             roi_size=(self.args.crop_size_w, self.args.crop_size_h, self.args.crop_size_d),
                                             num_samples=self.args.sample_per_mri if self.train else 1);
                else:
                    ret_transforms = self.crop_rand({'image1': mri1, 'image2': mri2,'mask': gt, 'gradient':gr, 'lbl' :np.ones_like(mri1)});
            
            ret_mri1 = None;
            ret_mri2 = None;
            ret_gt = None;
            ret_dt = None;

            for i in range(self.args.sample_per_mri):
                if self.args.deterministic is False:
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

                
                num_corrupted_patches = np.random.randint(1,5) if self.args.deterministic is False else 3;
                for _ in range(num_corrupted_patches):
                    mri2_c, heatmap = add_synthetic_lesion_wm(mri2_c, g, self.args.deterministic)
                    total_heatmap = torch.clamp(heatmap+total_heatmap, 0, 1);

                total_heatmap_thresh = torch.where(total_heatmap > 0.5, 1.0, 0.0);
                total_heatmap_thresh = torch.clamp(total_heatmap_thresh + gt_c, 0, 1);

                pos_dt = distance_transform_edt(np.where(total_heatmap_thresh.squeeze().numpy()==1, 0, 1));
                pos_dt = pos_dt/(np.max(pos_dt)+1e-4);

                neg_dt = distance_transform_edt(total_heatmap_thresh.squeeze().numpy()==1);
                neg_dt = neg_dt/(np.max(neg_dt)+1e-4);

                dt = pos_dt - neg_dt ;
                dt = torch.from_numpy(np.expand_dims(dt, axis = 0));
                
                if self.args.debug_train_data:
                    pos_cords = np.where(total_heatmap_thresh >0.0);
                    if len(pos_cords[0]) != 0:
                        r = np.random.randint(0,len(pos_cords[0]));
                        center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
                    else:
                        center=[mri2_c.shape[1]//2, mri2_c.shape[2]//2, mri2_c.shape[3]//2]
                    visualize_2d([mri1_c, mri2_c, total_heatmap_thresh], center);
                
                mri1_c = self.transforms(mri1_c);

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

            if self.args.debug_train_data:
                pos_cords = np.where(ret_gt == 1);
                if len(pos_cords[0]) != 0:
                    r = np.random.randint(0,len(pos_cords[0]));
                    center = [pos_cords[0][r], pos_cords[1][r],pos_cords[2][r]]
                else:
                    center=[mri1.shape[1]//2, mri1.shape[2]//2, mri1.shape[3]//2]
                visualize_2d([ret_mri1, ret_mri2, ret_gt, brainmask], center);

            return ret_mri1, ret_mri2, ret_gt, brainmask, patient_id, loc;
    def update_prediction(self, 
                          pred, 
                          patient_id, 
                          loc):
        """saves the prediction into the predefined tensor.
            location is set through 'loc' parameter

        
        Parameters
        ----------
        pred : np.ndarray
            prediction from the model for a particular patch of MRI.

        patient_id : str
            indicate for which testing example this prediction is.

        loc : int
            location of this predicted patch in the list of all patches for one particular example.

        """
        self.pred_data[patient_id][(loc[0].item())*self.step_w:(loc[0].item())*self.step_w + self.args.crop_size_w, 
                                (loc[1].item())*self.step_h:((loc[1].item()))*self.step_h + self.args.crop_size_h, 
                                (loc[2].item())*self.step_d:((loc[2].item()))*self.step_d + self.args.crop_size_d] = np.array(pred.squeeze().detach().cpu().numpy()).astype("int32");

    def calculate_metrics(self, simple = True):
        """After finishing all the prediction, we calculate F1, HD and dice metrics

        Parameters
        ----------
        simple : bool
            if true, only dice score is computer, otherwise F1 and HD are also computed.
        """
        ret = [];
        for k in tqdm(self.pred_data.keys()):
            if simple is True:
                dice = calculate_metric_percase(self.pred_data[k].squeeze(), self.gt_data[k].squeeze(), simple=simple);
            else:
                dice,hd,f1 = calculate_metric_percase(self.pred_data[k].squeeze(), self.gt_data[k].squeeze(), simple=simple);
            if np.sum(self.gt_data[k].squeeze()) > 0:
                ret.append(dice if simple is True else [dice, hd, f1]);
        return np.mean(ret) if simple is True else np.mean(np.array(ret), axis =0);

def get_loader_pretrain_miccai(args):
    """prepare train and test loader for self-supervised pretraining model

        Parameters
        ----------
        args : dict
            arguments.
    """
    train_mri, test_mri = pickle.load(open(os.path.join('cache_miccai-2016', f'train_test_split.dmp'), 'rb'));

    mri_dataset_train = MICCAI_PRETRAIN_Dataset(args, train_mri);
    train_loader = DataLoader(mri_dataset_train, 1, True, num_workers=args.num_workers, pin_memory=True);
    mri_dataset_test = MICCAI_PRETRAIN_Dataset(args, test_mri, train=False);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=args.num_workers, pin_memory=True);

    return train_loader, test_loader; 

def get_loader_miccai(args, fold):
    """prepare train and test loader for new lesion segmentation model

        Parameters
        ----------
        args : dict
            arguments.
    """
    with open(os.path.join('cache_miccai', f'fold{fold}.txt'), 'r') as f:
        train_ids = f.readline().rstrip();
        train_ids = train_ids.split(',');
        test_ids = f.readline().rstrip();
        test_ids = test_ids.split(',');
    train_ids =  [os.path.join('miccai-processed', t) for t in train_ids];
    test_ids = [os.path.join('miccai-processed', t) for t in test_ids];


    mri_dataset_train = MICCAI_Dataset(args, train_ids[:5], train=True);
    train_loader = DataLoader(mri_dataset_train, 1, True, num_workers=args.num_workers, pin_memory=True);
    mri_dataset_test = MICCAI_Dataset(args, test_ids[:5], train=False);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=args.num_workers, pin_memory=True);

    return train_loader, test_loader, mri_dataset_test; 

def visualize_2d(images, slice,):
    """display one slice of an MRI scan, for debugging purposes

        Parameters
        ----------
        args : dict
            arguments.
    """
    fig, ax = plt.subplots(len(images),3);
    for i,img in enumerate(images):
        img = img.squeeze();
        ax[i][0].imshow(img[slice[0], :,:], cmap='gray');
        ax[i][1].imshow(img[:,slice[1],:], cmap='gray');
        ax[i][2].imshow(img[:,:,slice[2]], cmap='gray');
    plt.show()

def inpaint_3d(img, 
               mask_g, 
               num_corrupted_patches, 
               deterministic = False):
    """remove part of an MRI scan for self-superivsed pretraining model

        Parameters
        ----------
        img : np.ndarray
            MRI scan patch

        mask_g : np.ndarray
            mask to take inpainting centers from

        num_corrupted_patches : int
            number of patches to curropt

        deterministic : bool
            if true, every call to this function yields the same results.
    """
    mri = img;

    _,h,w,d = mri.shape;

    cubes = [];
    for n in range(num_corrupted_patches):
        mask_cpy = deepcopy(mask_g);
        size_x = np.random.randint(5,15) if deterministic is False else 15;
        size_y = np.random.randint(5,20) if deterministic is False else 15;
        size_z = np.random.randint(5,20) if deterministic is False else 15;
        mask_cpy[:,:,:,d-size_z:] = 0;
        mask_cpy[:,:,:,:size_z+1] = 0;
        mask_cpy[:,:,w-size_y:,:] = 0;
        mask_cpy[:,:,:size_y+1,:] = 0;
        mask_cpy[:,h-size_x:,:,:] = 0;
        mask_cpy[:,:size_x+1,:,:] = 0;
        pos_cords = np.where(mask_cpy==1);

        if deterministic is False:
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

def add_synthetic_lesion_wm(img, 
                            mask_g, 
                            deterministic):
    """adds synthetic lesions to the MRI scan

        Parameters
        ----------
        img : np.ndarray
            MRI scan patch

        mask_g : np.ndarray
            mask to take inpainting centers from

        deterministic : bool
            if true, every call to this function yields the same results.
    """
    mri = img;

    _,h,w,d = mri.shape;

    mask_cpy = deepcopy(mask_g);
    size_x = np.random.randint(2,6) if deterministic is False else 3;
    size_y = size_x - np.random.randint(0,size_x-1) if deterministic is False else 3;
    size_z = size_x - np.random.randint(0,size_x-1) if deterministic is False else 3;
    mask_cpy[:,:,:,d-size_z:] = 0;
    mask_cpy[:,:,:,:size_z+1] = 0;
    mask_cpy[:,:,w-size_y:,:] = 0;
    mask_cpy[:,:,:size_y+1,:] = 0;
    mask_cpy[:,h-size_x:,:,:] = 0;
    mask_cpy[:,:size_x+1,:,:] = 0;
    pos_cords = np.where(mask_cpy==1);

    if deterministic is False:
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

    cube = GaussianSmooth(1.2, approx='erf')(cube);
    cube = cube / (torch.max(cube) + 1e-4);
    #================

    noise = (torch.ones((1,h,w,d), dtype=torch.uint8));
    final = (cube)*(noise);
    mri_after = (1-cube)*mri + final;
    
    return mri_after, cube;