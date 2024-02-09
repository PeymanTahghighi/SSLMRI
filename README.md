## Examples
### Detecting all changes
Here, our model detects all the changes from one MRI scan to the other. The first column represents the first MRI scan, and the second represents the second MRI scan. The third column indicates changes from the first MRI (first column) to the second MRI (second column). Blue regions indicate parts that have been removed from the first MRI to the second, and red regions show the parts that have been added from the first MRI to the second. Note that some of the changes were added programmatically for illustration purposes only.
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Example1.gif)
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Example2.gif)
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Example3.gif)
### Synthetic lesions
Here you can see exampes of synthetic lesions added to MRI scans.

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
To get same results are the paper first download [models](https://uofc-my.sharepoint.com/:f:/g/personal/peyman_tahghighi_ucalgary_ca/EjIxxA6_anpHr9L4RaX9lssB_yTIMH5AnPBPyxr09LhBwA?e=Au8FPr), place them in root folder, then run
```
python test.py
```
#### Detecting all changes
First download [samples](https://uofc-my.sharepoint.com/:f:/g/personal/peyman_tahghighi_ucalgary_ca/EkY72iGdbaRDlyYe8-yNzpQBwlVahqmmn-EjkFdEmY2cKQ?e=LV7nVG) and put them in the root folder. Then to see all the slices using matplotlib library use:
```
python test.py --segmentation
```
To create a gif from the prediction of the model, run:
```
python test.py --segmentation --gif
```
Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/New%20lesions/1_1.png)  |  ![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Examples/New%20lesions/1_2.png)
