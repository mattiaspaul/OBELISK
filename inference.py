from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import sys
import nibabel as nib
import scipy.io

import argparse

cuda_idx = 0

from utils import init_weights, countParam, augmentAffine, my_ohem, dice_coeff
from models import obeliskhybrid_tcia, obeliskhybrid_visceral


#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])

def main():
    #read/parse user command line input
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset ", dest="dataset", help="either tcia or visceral", default='tcia', required=True)
    parser.add_argument("-fold", dest="fold", help="number of training fold", default=1, required=True)
    parser.add_argument("-model", dest="model", help="filename of pytorch pth model", default='obeliskhybrid', required=True)
    parser.add_argument("-input", dest="input",  help="nii.gz CT volume to segment", required=True)
    parser.add_argument("-output", dest="output",  help="nii.gz label output prediction", default=None, required=True)
    parser.add_argument("-groundtruth", dest="groundtruth",  help="nii.gz groundtruth segmentation", default=None, required=False)

    options = parser.parse_args()
    d_options = vars(options)
    modelname = split_at(d_options['model'], '_', 1)[0]
    print('input CT image',d_options['input'],'\n   and model name',modelname,'for dataset',d_options['dataset'])
    
    img_val =  torch.from_numpy(nib.load(d_options['input']).get_data()).float().unsqueeze(0).unsqueeze(0)
    
    load_successful = False
    if((modelname=='obeliskhybrid')&(d_options['dataset']=='tcia')):
        net = obeliskhybrid_tcia(9) #has 8 anatomical foreground labels
        net.load_state_dict(torch.load(d_options['model']))
        img_val = img_val/1024.0 + 1.0 #scale data
        load_successful = True

    if((modelname=='obeliskhybrid')&(d_options['dataset']=='visceral')):
        img_val = img_val/1000.0
        _,_,D_in0,H_in0,W_in0 = img_val.size()
        with torch.no_grad():
            #subsample by factor of 2 (higher resolution in our original data)
            img_val = F.avg_pool3d(img_val,3,padding=1,stride=2)
        _,_,D_in1,H_in1,W_in1 = img_val.size()
        full_res = torch.Tensor([D_in1,H_in1,W_in1]).long()
        net = obeliskhybrid_visceral(8,full_res) #has 7 anatomical foreground labels
        net.load_state_dict(torch.load(d_options['model']))
        load_successful = True

    if(load_successful):
        print('read in model with',countParam(net),'parameters')
    else:
        print('model',modelname,'for dataset',d_options['dataset'],'not yet supported. exit()')
        exit()
       
    net.eval()

    if(torch.cuda.is_available()==1):
        print('using GPU acceleration')
        img_val = img_val.cuda()
        net.cuda()
    with torch.no_grad():
        predict = net(img_val)
        if(d_options['dataset']=='visceral'):
            predict = F.interpolate(predict,size=[D_in0,H_in0,W_in0], mode='trilinear', align_corners=False)

    argmax = torch.argmax(predict,dim=1)
    seg_img = nib.Nifti1Image(argmax.cpu().short().squeeze().numpy(), np.eye(4))
    print('saving nifti file with labels')
    nib.save(seg_img, d_options['output'])
       
    if d_options['groundtruth'] is not None:
        seg_val =  torch.from_numpy(nib.load(d_options['groundtruth']).get_data()).long().unsqueeze(0)
        dice = dice_coeff(argmax.cpu(), seg_val, predict.size(1)).numpy()
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('Dice validation:',dice)

        

if __name__ == '__main__':
    main()

