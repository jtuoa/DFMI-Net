from torch.utils.data import Dataset, DataLoader
import skimage.io as io
import torch
from PIL import Image


class FusionDataset(Dataset):
    '''
    Data loader: read from txt file
    '''
    def __init__(self, txt, transform = None, target_transform = None):
        super(FusionDataset,self).__init__()
        lists = open(txt, 'r')
        imgs = []
        for line in lists:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        ct, mr = self.imgs[index]
        img_ct = io.imread(ct)
        img_mr = io.imread(mr)
      
        img_ct = Image.fromarray(img_ct)
        img_mr = Image.fromarray(img_mr)
    
        if self.transform is not None:
            img_ct = self.transform(img_ct)
            img_mr = self.transform(img_mr)

        return img_ct, img_mr

    def __len__(self):
        return len(self.imgs)

        
# If the pixel value is below the threshold, this pixel is the background. 
BG_THRESHOLD = 17.0/256

def get_bg_mask(ct, mr):
    bg_mask = ((ct <= BG_THRESHOLD) & (mr <= BG_THRESHOLD))
    return bg_mask

def get_fg_mask(ct, mr):
    fg_mask = ((ct > BG_THRESHOLD) | (mr > BG_THRESHOLD))
    return fg_mask

def scale_image(img, fg_mask = None):
    if fg_mask is None:
        fg_mask = torch.ones(img.shape).type(torch.ByteTensor)
    fg_vals = torch.masked_select(img, fg_mask)
    minv = fg_vals.min()
    maxv = fg_vals.max()
    
    img = (img - minv)
    img = img / (maxv - minv)
    img[img > 1] = 1
    img[img < 0] = 0
    
    return img

	
def post_image(ct, mr, fused, chg_bg = True, inverse = True):
    '''
    args:
        2 input images, 1 fused image
    returns:
        scaled and masked fused image
    ''' 
    bg_mask = get_bg_mask(ct, mr)
    fg_mask = get_fg_mask(ct, mr)
    
    bg_mask = bg_mask.cpu().numpy()
    
    r = scale_image(fused, fg_mask)
    if inverse:
        r = 1 - r
    if chg_bg: 
        r[bg_mask] = 0

    return r



	
