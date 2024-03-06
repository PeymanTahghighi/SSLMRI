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

