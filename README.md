### Examples
## Detecting all changes
Here, our model detects all the changes from one MRI scan to the other. The first column represents the first MRI scan, and the second represents the second MRI scan. The third column indicates changes from the first MRI (first column) to the second MRI (second column). Blue regions indicate parts that have been removed from the first MRI to the second, and red regions show the parts that have been added from the first MRI to the second. Note that some of the changes were added programmatically for illustration purposes only.
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Example1.gif)
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Example2.gif)
![](https://github.com/PeymanTahghighi/SSLMRI/blob/master/Example3.gif)
## Synthetic lesions
Here you can see exampes of synthetic lesions added to MRI scans.

### Usage
To pretrain model using self-supervised learning you have to cache MRI testing dataset the first time. Hence run code like this ONLY ONCE
```
python run.py --pretraining --cache-mri-data
```
Then, after caching data for future runs you can use
```
python run.py --pretraining
```

To train new lesion segmentation model, run as below
```
python run.py
```


