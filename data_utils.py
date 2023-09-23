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
from monai.transforms import Rand3DElastic, Resize, RandGaussianSmooth, OneOf, RandGibbsNoise, RandGaussianNoise, GaussianSmooth, NormalizeIntensity, RandCropByPosNegLabeld, GibbsNoise
from scipy.ndimage import convolve
from tqdm import tqdm
import math

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
    
    mri_dataset_test = MRI_Dataset_3D(test_mri,cache=True);
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

class MRI_Dataset_3D(Dataset):
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

            for i in range(config.hyperparameters['sample_per_mri'] if self.cache is False else 1):
                if config.hyperparameters['deterministic'] is False:
                    ret_transforms = self.crop({'image': mrimage,'gradient': g, 'mask': mask});
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
                    mrimage_noisy, heatmap, noise, center = add_synthetic_lesion_3d(mrimage_noisy, g_c)
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

def get_loader(fold):
    
    train_mri, test_mri = pickle.load(open(f'cache/{fold}.fold', 'rb'));

    mri_dataset_train = MRI_Dataset_3D(train_mri[:5]);
    train_loader = DataLoader(mri_dataset_train, 1, True, num_workers=8, pin_memory=True);
    test_mri = glob(os.path.join('cache',f'{fold}','*.tstd'));
    mri_dataset_test = MRI_Dataset_3D(test_mri[:5], train=False);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=8, pin_memory=True);

    return train_loader, test_loader;   

def standardize(img):
    img = img - np.min(img);
    img = (img / np.max(img))*255;
    return img;

# def cache_lesion_center_points(img):
#     if type(img) is str:
#         mri = cv2.imread(img, cv2.IMREAD_GRAYSCALE);
#     else:
#         mri = img;
#     mri_smoothed = gaussian(mri, 1);
#     mri = mri/255;
#     h,w = mri.shape;
#     mri_thresh = threshold_otsu(mri);


#     sobel_mag = np.sqrt(sum([sobel(mri_smoothed, axis=i)**2
#                          for i in range(mri.ndim)]) / mri.ndim)
    
#     sobel_mag = ((mri>mri_thresh)*(sobel_mag<0.1)).astype("uint8");
#     pos_cords = np.array(range(sobel_mag.shape[0]*sobel_mag.shape[1])).reshape(sobel_mag.shape[0], sobel_mag.shape[1]);

#     pos_cords = pos_cords[sobel_mag==1];
#     low = 35;
#     high = 40;
#     size = np.random.randint(low,high);

#     num_lesions = 10;
    
#     pos_cords_temp = copy(pos_cords);
#     cached_lesion_center = [];
#     while(num_lesions >= 0):
#         if len(pos_cords_temp) <=0:
#             low-=1;
#             size = np.random.randint(low,size);
#             pos_cords_temp = copy(pos_cords);
#         r = np.random.randint(0, len(pos_cords_temp));
        
#         point = [math.floor(pos_cords_temp[r]/h), pos_cords_temp[r]%h]

#         start_h = max(point[0]-size, 0);
#         end_h = min(point[0]+size, h);

#         start_w = max(point[1]-size,0);
#         end_w = min(point[1]+size,h);
#         patch = sobel_mag[start_h:end_h, start_w:end_w];
#         s = np.sum(patch);
#         if s >= (size*2 * size*2) > 0.78:
#             cached_lesion_center.append([point[0], point[1], size]);
#             pos_cords_temp = np.delete(pos_cords_temp, r);
#             num_lesions -= 1;
#         else:
#             pos_cords_temp = np.delete(pos_cords_temp, r);
    
#     debug_lesion_location = False;
#     if debug_lesion_location:
#         sobel_mag = (sobel_mag*255).astype("uint8")
#         sobel_mag = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2RGB);
#         for i in range(len(cached_lesion_center)):
#                 sobel_mag_marked = deepcopy(sobel_mag);
#                 sobel_mag_marked = cv2.circle(sobel_mag_marked, (int(cached_lesion_center[i][1]), int(cached_lesion_center[i][0])), cached_lesion_center[i][2], (255,255,0), -1);
#                 fix,ax = plt.subplots(1,3);
#                 ax[0].imshow(mri, cmap = 'gray');
#                 ax[1].imshow(sobel_mag);
#                 ax[2].imshow(sobel_mag_marked);
#                 plt.show();

#     return cached_lesion_center;

# def add_synthetic_lesion_2d(img, center, size):
#     if type(img) is str:
#         mri = cv2.imread(img, cv2.IMREAD_GRAYSCALE);
#     else:
#         mri = img;
    
#     mri = mri/255;
#     h,w = mri.shape;
 
#     #shape
#     ellipse = np.zeros((h,w,), dtype=np.uint8);
#     ellipse = cv2.ellipse(ellipse, center=(int(center[1]), int(center[0])), axes= (size,size), angle=0, startAngle=0, endAngle=360, color= (255,255,255), thickness= -1);
#     els = iaa.Sequential([iaa.ElasticTransformation(alpha=(80,120), sigma=(5,10))])
#     ellipse = els(images = ellipse);
#     ellipse_thresh = (ellipse>0)
#     ellipse_thresh = gaussian(ellipse_thresh, 7);
#     #================

#     downsample = np.random.randint(10,15);

#     #noise
#     noise = np.random.normal(size=(int(h/downsample),int(w/downsample)), loc=1.0, scale=1.0);
#     noise = np.clip(noise, a_min = 0.2, a_max = 6);
#     noise = cv2.resize(noise, (h,w), interpolation=cv2.INTER_LINEAR)
#     noise = gaussian(noise, 7);
#     #================

#     final = (ellipse_thresh)*(mri*noise);

#     mri_after = (1-ellipse_thresh)*mri + final;

#     debug_output = False;
#     if debug_output is True:
#         diff = mri - mri_after;
#         fix, ax = plt.subplots(1,3);
#         ax[0].imshow(mri);
#         ax[1].imshow(diff);
#         plt.show();
    
#     mri_after = np.clip(mri_after, 0, 1);
#     mri_after = (mri_after*255).astype("uint8")
#     return mri_after, ellipse_thresh;

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

def add_synthetic_lesion_3d(img, mask = None):
    
    mri = img;

    _,h,w,d = mri.shape;

    mask_cpy = deepcopy(mask);
    size = np.random.randint(15,25) if config.hyperparameters['deterministic'] is False else 15;
    mask_cpy[:,:,:,d-size:] = 0;
    mask_cpy[:,:,:,:size+1] = 0;
    mask_cpy[:,:,w-size:,:] = 0;
    mask_cpy[:,:,:size+1,:] = 0;
    mask_cpy[:,h-size:,:,:] = 0;
    mask_cpy[:,:size+1,:,:] = 0;
    pos_cords = np.where(mask_cpy==1);
    r = np.random.randint(0,len(pos_cords[0]));
    center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
 
    #shape
    cube = np.zeros((1,h,w,d), dtype=np.uint8);
    cube[:,max(center[0]-size,0):min(center[0]+size, h), max(center[1]-size,0):min(center[1]+size,w), max(center[2]-size,0):min(center[2]+size,d)] = 1;

    #cube = transform(cube);
    cube_thresh = (cube>0)

    cube_thresh = GaussianSmooth(7, approx='erf')(cube_thresh);
    cube_thresh = cube_thresh / (torch.max(cube_thresh) + 1e-4);
    #================

    noise = GaussianSmooth(7)(mri);
    final = (cube_thresh)*(noise);
    noise = GaussianSmooth(7)(mask.float());
    mri_after = (1-cube_thresh)*mri + final;
    
    mri_after = torch.clip(mri_after, 0, 1);
    #mri_after = (mri_after*255).astype("uint8")
    #visualize_2d(mri_after, cube_thresh, slice=center[0:]);
    return mri_after, cube_thresh, noise, center;

# def inpaint_3d(img):
#     if type(img) is str:
#         mri = cv2.imread(img, cv2.IMREAD_GRAYSCALE);
#     else:
#         mri = img;
    
#     mri = mri/255;
#     mri = torch.from_numpy(mri);
#     h,w,d = mri.shape;

#     thresh = threshold_otsu(img);
#     thresh = img > thresh;

#     pos_cords = np.where(thresh==1);
#     r = np.random.randint(0,len(pos_cords[0]));
#     center = [pos_cords[0][r], pos_cords[1][r],pos_cords[2][r]]
#     size = [np.random.randint(20,50), np.random.randint(20,50), np.random.randint(20,50)]
 
#     #shape
#     cube = np.zeros((1,h,w,d), dtype=np.uint8);
#     cube[:,center[0]-size[0]:center[0]+size[0], center[1]-size[1]:center[1]+size[1], center[2]-size[2]:center[2]+size[2]] = 255;

#     transform = RandomElasticDeformation(
#         num_control_points=np.random.randint(20,30),
#         locked_borders=2,
#         max_displacement=np.random.randint(40,60)
#     )

#     cube = transform(cube);
#     cube_thresh = (cube>0)
#     cube_thresh = GaussianSmooth(7)(cube_thresh);
#     #================

#     downsample = np.random.randint(10,15);

#     # #noise
#     noise = np.random.normal(size=(1,int(h/downsample),int(w/downsample), int(d/downsample)), loc=2, scale=4.0);
#     noise = Resize((h,w,d))(noise);
#     noise = GaussianSmooth(7)(noise);
#     #================

#     final = (cube_thresh)*(mri*noise);

#     mri_after = (1-cube_thresh)*mri + final;
    
#     mri_after = np.clip(mri_after, 0, 1);
#     mri_after = (mri_after*255).astype("uint8")
#     #visualize_2d(mri_after, slice=center);
#     return mri_after, cube_thresh;

# def inpaint(img):
    
#     if type(img) is str:
#         mri = cv2.imread(img, cv2.IMREAD_GRAYSCALE);
#     else:
#         mri = img;
#     mri_smoothed = gaussian(mri, 1);
#     mri = mri/255;
#     h,w = mri.shape;
#     mri_thresh = threshold_otsu(mri);


#     sobel_mag = np.sqrt(sum([sobel(mri_smoothed, axis=i)**2
#                          for i in range(mri.ndim)]) / mri.ndim)
    
#     sobel_mag = ((mri>mri_thresh)*(sobel_mag<0.1)).astype("uint8");
#     edges = np.sqrt(sum([sobel(sobel_mag, axis=i)**2
#                          for i in range(mri.ndim)]) / mri.ndim)
#     edges = edges > threshold_otsu(edges);
#     pos_cords = np.array(range(edges.shape[0]*edges.shape[1])).reshape(edges.shape[0], edges.shape[1]);

#     pos_cords = pos_cords[edges==1];
#     r = np.random.randint(0, len(pos_cords));

#     top_left = [math.floor(pos_cords[r]/h), pos_cords[r]%h];

#     mri_smoothed = gaussian(mri, 21);
 
#     #shape
#     rh = np.random.randint(100,150);
#     rw = np.random.randint(100,150);
#     rectangle = np.zeros((h,w,), dtype=np.uint8);
#     rectangle = cv2.rectangle(rectangle, pt1=(top_left[1],top_left[0]), pt2=(top_left[1]+rh ,top_left[0]+rw), color= (255,255,255), thickness= -1);
#     edges = cv2.rectangle((edges*255).astype("uint8"), pt1=(top_left[1],top_left[0]), pt2=(top_left[1]+rh ,top_left[0]+rw), color= (255,255,255), thickness= -1);
#     rectangle_thresh = (rectangle>0)
#     rectangle_thresh = gaussian(rectangle, 7);
#     #================


#     final = (rectangle_thresh)*(mri_smoothed);

#     mri_after = (1-rectangle_thresh)*mri + final;

#     debug_output = True;
#     if debug_output is True:
#         fix, ax = plt.subplots(1,3);
#         ax[0].imshow(mri, cmap='gray');
#         ax[2].imshow(mri_after, cmap='gray');
#         ax[1].imshow(edges, cmap='gray');
#         plt.show();
    
#     mri_after = np.clip(mri_after, 0, 1);
#     mri_after = (mri_after*255).astype("uint8")
#     return mri_after;

# def cache_lesion_locations():
#     all_imgs = glob('dataset/*.png');
#     for img_path in all_imgs:
#         file_name = os.path.basename(img_path);
#         file_name = file_name[:file_name.rfind('.')];
#         ret = cache_lesion_center_points(img_path);
#         pickle.dump(ret, open(f'dataset/{file_name}.dmp', 'wb'));

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