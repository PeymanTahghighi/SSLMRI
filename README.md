## Usage
### Downloading
Please install [PyTorch](https://pytorch.org/) and download the [MICCAI-21](https://portal.fli-iam.irisa.fr/msseg-2/) dataset for new lesion segmentation and [MICCAI-16](https://portal.fli-iam.irisa.fr/msseg-challenge/english-msseg-data/) for self-supervised pre-training. Then, run standard provided preprocessing on them and put them in under the 'miccai-processed' folder for the MICCAI-21 (MSSEG-2) dataset and the 'miccai-2016' folder for the MICCAI-16 (MSSEG) dataset. Finally, download model checkpoints for both self-supervised pre-training and new lesion segmentation from [here](https://fastupload.io/lQpetkKWaVIzFKq/file) and extract it into the root directoy.
### Training
#### Self-supervised pretraining
To pretrain model using self-supervised learning you have to cache MRI testing dataset the first time. Hence run code like below **ONLY ONCE**
```
python run.py --pretraining --cache-mri-data
```
Then, after caching data for future runs you can use
```
python run.py --pretraining
```
#### New lesion segmentation model
To train new lesion segmentation model without self-supervised pre-trained weights, run as below:
```
python run.py --f 0 --bl-multiplier 10
```
where 'f' specifies the fold you wish the model to be trained on and 'bl-multiplier' determine boundary loss coefficient.

To train new lesion segmentation model with self-supervised pre-trained weights, run as below:
```
python run.py --pre-trained --f 0 --bl-multiplier 10
```

### Testing
#### New lesion segmentation
To get same results as the paper, run as below:
```
python test.py
```
The results of each fold will then be save under 'Results' folder and the average results accross five-fold will be printed.
