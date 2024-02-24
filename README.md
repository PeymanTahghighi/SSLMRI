## Examples
### Self-supervised pre-training results
Here, outputs of the self-supervised pre-training are shown, which highlights changes from one MRI scan to the other. Note that we did not quantify this in our paper, and it is only based on the observation. We did not claim this in the paper.
The first column represents the first MRI scan, and the second represents the second MRI scan. The third column indicates changes from the first MRI (first column) to the second MRI (second column). Blue regions indicate parts that have been removed from the first MRI to the second, and red regions show the parts that have been added from the first MRI to the second. Note that some of the changes were added programmatically for illustration purposes only.
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/changes/Example1.gif)
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/changes/Example2.gif)
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/changes/Example3.gif)
### Synthetic lesions
Here, you can see examples of synthetic lesions added to MRI scans. The first column shows the original MRI scan patch before adding a synthetic lesion. The second column shows an MRI scan after adding a synthetic lesion. The third column shows the mask of the added lesion to the original MRI scan patch.
Original MRI Patch             |  MRI patch with added lesion | Mask
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/New%20lesions/1_1.png)  |  ![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/New%20lesions/1_2.png)|  ![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/New%20lesions/1_3.png) 
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/New%20lesions/2_1.png)  |  ![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/New%20lesions/2_2.png)|  ![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/New%20lesions/2_3.png)
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/New%20lesions/3_1.png)  |  ![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/New%20lesions/3_2.png)|  ![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/New%20lesions/3_3.png)
## Usage
### Downloading
Please install [PyTorch](https://pytorch.org/) and download the [MICCAI-21](https://portal.fli-iam.irisa.fr/msseg-2/) dataset. Then run standard provided preprocessing on them and put them in under 'miccai-processed' folder.
### Self-supervised pretraining
To pretrain model using self-supervised learning you have to cache MRI testing dataset the first time. Hence run code like this ONLY ONCE
```
python run.py --pretraining --cache-mri-data
```
Then, after caching data for future runs you can use
```
python run.py --pretraining
```
### New lesion segmentation model
To train new lesion segmentation model, run as below
```
python run.py
```
### Testing
#### New lesion segmentation
To get same results are the paper first download [models](https://file.io/cS1dZg25VqlW), place them in root folder, then run
```
python test.py
```
#### Outputs of self-supervised pre-training
First download [samples](https://file.io/dFi2Br52YfH6) and put them in the root folder. Then to visualize the outputs using matplotlib library use:
```
python test.py --segmentation
```
To create a gif from the prediction of the model, run:
```
python test.py --segmentation --gif
```

