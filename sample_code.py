import argparse
import numpy as np
import skimage.io as io
import warnings

import torchvision, torch
import torch.nn as nn
from torchvision import transforms, utils

from network.FWNet import FWNet
from utils import *

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss")
parser.add_argument('--test_dir', '-t', type=str, required=True, help='Input images directory.')
parser.add_argument('--out_dir', '-o', type=str, required=True, help='Path to output directory.')
parser.add_argument('--weights', '-w', type=str, required=True, help='Path to the trained CNN weights.')
args = parser.parse_args()


# load inference weights:
net = FWNet(1, 2)
net.to(device)
net.load_state_dict(torch.load(args.weights)) #re-train
net.eval()

# load test dataset:
img_transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
test_data = FusionDataset(txt = args.test_dir, transform = img_transform)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 1, num_workers = 1)

#run prediction:
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        img_ct, img_mr = data
        img_ct, img_mr = img_ct.to(device), img_mr.to(device)
        
        img_fusion, outputs = net(torch.cat((img_ct, img_mr), dim = 1).to(device))
        
        # scale image
        r = post_image(img_ct[0][0], img_mr[0][0], img_fusion[0][0], chg_bg=True, inverse=False).cpu()
        
        # output: fused img
        io.imsave('{}/fuse_{:04}.png'.format(args.out_dir, i+40), np.array(r * 255, dtype = 'uint8'))

print('inference done!')






