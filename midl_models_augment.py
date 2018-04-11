import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import time
import os
import sys
from torch.autograd import Variable

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0.0)
    if isinstance(m, nn.Conv1d):
        init.xavier_normal(m.weight, gain=np.sqrt(2))

#defines binary 3D sampling layout for 1024 filter kernel elements
class OBELISKNet(nn.Module):
    def __init__(self):
        super(OBELISKNet, self).__init__()
        self.xyz_off = nn.Linear(1024,6) #trainable parameters
    
    def forward(self, input, xyz_sample, A):
        #input = batch of 3D images (pre-smoothed)
        #xyz_sample = spatial sampling locations (grid or scattered)
        #A affine matrices (can be different for each location)
        B,Feats,_ = xyz_sample.size()
        xyz_sample = xyz_sample.view(B*Feats,3)
        
        #matrix multiplication with affine matrix for online augmentation
        xyz1 = torch.bmm(A,self.xyz_off.weight[:3,:].unsqueeze(0).repeat(B*Feats,1,1))
        x_feat1 = xyz_sample[:,2:3] + xyz1[:,0,:]
        y_feat1 = xyz_sample[:,1:2] + xyz1[:,1,:]
        z_feat1 = xyz_sample[:,0:1] + xyz1[:,2,:]
        xyz_grid1 = torch.cat((z_feat1.unsqueeze(2),y_feat1.unsqueeze(2),x_feat1.unsqueeze(2)),2).view(B,-1,1024,1,3)
        
        xyz2 = torch.bmm(A,self.xyz_off.weight[3:,:].unsqueeze(0).repeat(B*Feats,1,1))
        x_feat2 = xyz_sample[:,2:3] + xyz2[:,0,:]
        y_feat2 = xyz_sample[:,1:2] + xyz2[:,1,:]
        z_feat2 = xyz_sample[:,0:1] + xyz2[:,2,:]
        xyz_grid2 = torch.cat((z_feat2.unsqueeze(2),y_feat2.unsqueeze(2),x_feat2.unsqueeze(2)),2).view(B,-1,1024,1,3)

        #here happens the magic (requires pytorch v0.4 or higher)!
        features = (F.grid_sample(input,xyz_grid1)-F.grid_sample(input,xyz_grid2)).view(B,-1,1024)
        
        return features


# too simple to require comments 
class LinearNet(nn.Module):
    def __init__(self, num_labels):
        super(LinearNet, self).__init__()
        
        self.LIN1 = nn.Conv1d(1024, 256,1,bias=False,groups=4) #grouped convolutions
        self.BN1 = nn.BatchNorm1d(256)
        #self.DRO1 = nn.Dropout(0.2)
        self.LIN2 = nn.Conv1d(256, 128,1,bias=False)
        self.BN2 = nn.BatchNorm1d(128)
        
        self.LIN3 = nn.Conv1d(128, 128,1,bias=False)
        self.BN3 = nn.BatchNorm1d(128)
        
        self.LIN4 = nn.Linear(128, num_labels)
    
    def forward(self, input):
        x1 = F.relu(self.BN1(self.LIN1(input.unsqueeze(2))))
        x2 = F.relu(self.BN2(self.LIN2(x1)))
        x3 = F.relu(self.BN3(self.LIN3(x2)))

        return self.LIN4(x3.squeeze(2))
    
#slightly more complicated linear 1x1 densenet architecture
class DenseLinearNet(nn.Module):
    def __init__(self, num_labels):
        super(DenseLinearNet, self).__init__()
        self.LIN1 = nn.Conv1d(1024, 256, 1, bias=False, groups=4) #grouped convolutions
        self.BN1 = nn.BatchNorm1d(256)
        self.LIN2 = nn.Conv1d(256, 128, 1, bias=False)
        self.BN2 = nn.BatchNorm1d(128)
        
        self.LIN3a = nn.Conv1d(128, 32, 1,bias=False)
        self.BN3a = nn.BatchNorm1d(128+32)
        self.LIN3b = nn.Conv1d(128+32, 32, 1,bias=False)
        self.BN3b = nn.BatchNorm1d(128+64)
        self.LIN3c = nn.Conv1d(128+64, 32, 1,bias=False)
        self.BN3c = nn.BatchNorm1d(128+96)
        self.LIN3d = nn.Conv1d(128+96, 32, 1,bias=False)
        self.BN3d = nn.BatchNorm1d(256)
        
        self.LIN4 = nn.Linear(256, num_labels)
    
    def forward(self, input):
        x1 = F.relu(self.BN1(self.LIN1(input.unsqueeze(2))))
        x2 = self.BN2(self.LIN2(x1))
        
        x3a = torch.cat((x2,F.relu(self.LIN3a(x2))),dim=1)
        x3b = torch.cat((x3a,F.relu(self.LIN3b(self.BN3a(x3a)))),dim=1)
        x3c = torch.cat((x3b,F.relu(self.LIN3c(self.BN3b(x3b)))),dim=1)
        x3d = torch.cat((x3c,F.relu(self.LIN3d(self.BN3c(x3c)))),dim=1)

        return self.LIN4(self.BN3d(x3d).squeeze(2))


def netCount(net):
    count = 0
    for i, imod in enumerate(net.named_children()):
        if isinstance(imod[1], nn.Conv1d):
            count += (torch.numel(imod[1].weight))
        if isinstance(imod[1], nn.Linear):
            count += (torch.numel(imod[1].weight))
    print(('count',count))