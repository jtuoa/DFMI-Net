# DFMI-Net

Multiview 3-D Echocardiography Image Fusion with Mutual Information Neural Estimation

## Requirements
* Python 3.8.5
* Torch 1.6.0
* Torchvision 0.2.2
* OpenCV 4.1.2.30
* Numpy 1.19.1
* cuDNN 7.6.5
* NVIDIA GTX 1080 TI
```
pip install -r requirements.txt
```

## Usage
### Dataset
Awaiting for permissions to upload the medical image dataset.

### Pretrained model
The pretrained model checkpoints can be found in the checkpoints folder on Google Drive: [chkpts](https://drive.google.com/file/d/10waUwCJ3Ol-Ms1R-R1-xOHl9ODHqR8Qt/view?usp=sharing)

### Inference
Sample code for inference using the DFMI-Net model
```
python sample_code.py -t ./input/test_list.txt -o ./output -w ./model/ckpt.pt 
```

## Reference
If you find this work useful in your research, please cite:
```
@inproceedings{Ting2020dfminet,
  title = {Multiview {3-D} Echocardiography Image Fusion with Mutual Information Neural Estimation},
  author = {Ting, J. and Punithakumar, K. and Ray, N.},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine},
  pages = {765-771},
  year = {2020},
}
```
