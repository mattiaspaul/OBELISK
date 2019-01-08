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

foldNo = int(sys.argv[1])


from utils import init_weights, countParam, augmentAffine, my_ohem, dice_coeff
from models import *#obeliskhybrid_tcia, obeliskhybrid_visceral

pth = torch.load('midl_augment_net_obelisk'+str(foldNo)+'.pth')
full_res = torch.Tensor([156,115,160]).long(); net = obelisk_visceral(8,full_res)
pth_off = torch.load('midl_augment_offset_obelisk'+str(foldNo)+'.pth')
offset1 = pth_off['xyz_off.weight'].permute(1,0).contiguous().view(1,1024,1,2,3).flip(4)
#>>> print(pth_off['xyz_off.weight'].size())
#torch.Size([6, 1024])
#>>> print(net.offset1.size())
#torch.Size([1, 1024, 1, 2, 3])
pth.update({'offset1':offset1})
pth.move_to_end('offset1', last=False)

pth.update({'LIN1.weight':pth['LIN1.weight'].unsqueeze(3).unsqueeze(4)})
pth.update({'LIN2.weight':pth['LIN2.weight'].unsqueeze(3).unsqueeze(4)})
pth.update({'LIN3a.weight':pth['LIN3a.weight'].unsqueeze(3).unsqueeze(4)})
pth.update({'LIN3b.weight':pth['LIN3b.weight'].unsqueeze(3).unsqueeze(4)})
pth.update({'LIN3c.weight':pth['LIN3c.weight'].unsqueeze(3).unsqueeze(4)})
pth.update({'LIN3d.weight':pth['LIN3d.weight'].unsqueeze(3).unsqueeze(4)})
pth.update({'LIN4.weight':pth['LIN4.weight'].unsqueeze(2).unsqueeze(3).unsqueeze(4)})
net.load_state_dict(pth)

torch.save(net.state_dict(), 'obelisk_visceral_fold'+str(foldNo)+'.pth')

