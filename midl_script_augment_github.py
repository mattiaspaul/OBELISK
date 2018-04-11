from __future__ import print_function
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
import nibabel as nib

#TO DO, implement your own dataloader for scans and segmentations
#from midl_load_data import *
#result could be pre-smoothed using the following:
#pre-smooth images with approx. Gaussian kernel (best on GPU)
    #imgTestFilt = F.avg_pool3d(imgTest.cuda(),5,stride=1,padding=2)
    #imgTestFilt = F.avg_pool3d(imgTestFilt,5,stride=1,padding=2)
    #imgTestFilt = F.avg_pool3d(imgTestFilt,3,stride=1,padding=1).data.cpu()

from midl_models_augment import *

#call for multiple folds
fold = sys.argv[1]
fold = int(fold)

#capture command line output to log-file (remove is not required)
old_stdout = sys.stdout
log_file = open("midl_script_augment"+str(fold)+".log","w")
sys.stdout = log_file
print("midl_script_augment.log")


print(('fold number ',fold))

#online hard example mining
class NLL_OHEM(torch.nn.NLLLoss):                                                     
    """ Online hard example mining. 
    Needs input from nn.LogSoftmax() """                                             
                                                                                   
    def __init__(self, ratio):      
        super(NLL_OHEM, self).__init__(None, True)                                 
        self.ratio = ratio                                                         
                                                                                   
    def forward(self, x, y, ratio=None):                                           
        if ratio is not None:                                                      
            self.ratio = ratio                                                     
        num_inst = x.size(0)                                                       
        num_hns = int(self.ratio * num_inst)                                       
        x_ = x.clone()                                                             
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()              
        for idx, label in enumerate(y.data):                                       
            inst_losses[idx] = -x_.data[idx, label]                                 
        #loss_incs = -x_.sum(1)                                                    
        _, idxs = inst_losses.topk(num_hns)                                        
        x_hn = x.index_select(0, idxs)                                             
        y_hn = y.index_select(0, idxs)                                             
        return torch.nn.functional.nll_loss(x_hn, y_hn) 


#expected call for data loader
#imgs_filt,segs,imgTestFilt,segTest,imgTest = load_data(fold)


#validation
def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (torch.mean(iflat) + torch.mean(tflat))
    return dice


#prepare grid for sampling and test scan evaluation
D = 312; H = 230; W = 320; num_labels = 8;

x = torch.arange(D)
y = torch.arange(H)
z = torch.arange(W)

# dense grid coordinates (3d)
x_grid = Variable(x.view(-1,1,1).repeat(1,y.size(0),z.size(0)))
y_grid = Variable(y.view(1,-1,1).repeat(x.size(0),1,z.size(0)))
z_grid = Variable(z.view(1,1,-1).repeat(x.size(0),y.size(0),1))

xyz_grid = torch.stack((z_grid,y_grid,x_grid),0).view(1,3,D,H,W)
print(xyz_grid.size())

x = torch.linspace(-.9,.9,40)
y = torch.linspace(-.9,.9,40)
z = torch.linspace(-.9,.9,40)

# strided grid locations for test features
x_grid_test = Variable(x.view(-1,1,1).repeat(1,y.size(0),z.size(0)))
y_grid_test = Variable(y.view(1,-1,1).repeat(x.size(0),1,z.size(0)))
z_grid_test = Variable(z.view(1,1,-1).repeat(x.size(0),y.size(0),1))

xyz_sample_test = torch.stack((z_grid_test,y_grid_test,x_grid_test),3).view(1,40,-1,3)

#important segmentation labels have to be sampled using nearest neighbour interpolation
#therefore we first generate a dense 3D grid that can be rounded after scattered sampling
sampled_coords = F.grid_sample(xyz_grid,xyz_sample_test[0:1,:,:,:].unsqueeze(2)).round().long().squeeze()
sampled_idx = sampled_coords[2,:,:]*H*W + sampled_coords[1,:,:]*W + sampled_coords[0,:,:]
seg1 = segTest.view(-1)
labelTest = seg1[sampled_idx.view(-1)].view(40,1600)


print(labelTest.size())

#network initialisation and learning parameters

net = DenseLinearNet(num_labels)
#net = LinearNet(num_labels)
net.apply(init_weights)
netCount(net)

obelisk = OBELISKNet()
print(obelisk.xyz_off.weight.size())
obelisk.xyz_off.weight.data = torch.randn(6,1024)*0.05

obelisk.cuda()
net.cuda()

criterion = NLL_OHEM(.25)#0.25 
batch_size = 256#256
epoch_size = 64
params = list(obelisk.parameters()) + list(net.parameters())

optimizer = optim.Adam(params, lr=0.002)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.96)

run_loss = np.zeros(50)

#for loop over iterations and epochs
for epoch in range(50):
    
    xyz_sample = (torch.rand(9,batch_size,epoch_size,3)*2.0-1.0)
    labelBatch = torch.zeros(9,batch_size,epoch_size).long()
    for i in range(9):
        #again we need to make some extra effort to get nearest neighbour sampling of GT labels
        sampled_coords = F.grid_sample(xyz_grid,xyz_sample[i:i+1,:,:].unsqueeze(2)).round().long().squeeze()
        sampled_idx = sampled_coords[2,:,:]*H*W + sampled_coords[1,:,:]*W + sampled_coords[0,:,:]
        seg1 = segs[i:i+1,:,:,:].view(-1)
        labelBatch[i,:,:] = seg1[sampled_idx.view(-1)].view(batch_size,epoch_size)
    net.cuda()
    obelisk.cuda()
    net.train()
    obelisk.train()
    run_loss[epoch] = 0.0
    t1 = 0.0
    #we use sub-iterations of 3 images each to further reduce memory demand
    sub_idx = torch.randperm(9).view(3,3)
    for sub in range(3):
        imgs_cuda = Variable(imgs_filt[sub_idx[:,sub],:,:,:,:]/500.0).cuda()
        xyz_cuda = Variable(xyz_sample[sub_idx[:,sub],:,:,:]).cuda()
        label_cuda = Variable(labelBatch[sub_idx[:,sub],:,:]).cuda()
        
        t0 = time.time()
        for iter in range(epoch_size):
            A1 = torch.Tensor([[1,0,0],[0,1,0],[0,0,1]]).view(1,3,3)
            affine_cuda = Variable(A1+torch.randn(torch.numel(xyz_cuda[:,:,iter,0]),3,3)*0.2).cuda()
            optimizer.zero_grad() 
            #forward path and loss 
            featBatch = obelisk(imgs_cuda,xyz_cuda[:,:,iter,:],affine_cuda).view(-1,1024)
            output = net(featBatch)
            loss = criterion(F.log_softmax(output,dim=1),label_cuda[:,:,iter].view(-1))
            #backward path and weight updates
            loss.backward()
            run_loss[epoch] += loss.item()
            optimizer.step()
            del loss
            del output
            del featBatch
            torch.cuda.empty_cache()
        t1 += (time.time() - t0)
        del imgs_cuda
        del xyz_cuda
        del label_cuda
        torch.cuda.empty_cache()
        
    scheduler.step()
    #validation of held-out test image
    net.eval()
    obelisk.eval()
    imgs_cuda = Variable(imgTestFilt[0:1,:,:,:]/500.0).cuda()
    xyz_cuda = Variable(xyz_sample_test).cuda()

    output_test = torch.zeros(40,1600,num_labels)
    for i in range(40):
        torch.no_grad():
            A1 = torch.Tensor([[1,0,0],[0,1,0],[0,0,1]]).view(1,3,3)
            affine_cuda = (A1+torch.zeros(torch.numel(xyz_cuda[:,i,:,0]),3,3)).cuda()
            #affine matrix should now be identity
            featTest = obelisk(imgs_cuda,xyz_cuda[:,i,:,:],affine_cuda).view(-1,1024)
            output_test[i,:,:] = net(featTest).cpu().data
    _, argmax = torch.max(output_test,dim=2)
    
    dice_all = dice_coeff(argmax, labelTest, 8)
    del featTest
    del imgs_cuda
    del xyz_cuda
    torch.cuda.empty_cache()

    #print some feedback information
    print('epoch',epoch,'dice_avg','%.3f'%(torch.mean(dice_all)*100.0),'time train','%.3f'%t1,'loss','%.3f'%(run_loss[epoch]),'stddev','%.3f'%(torch.std(obelisk.xyz_off.weight.data)))
    
    #move to cpu for storage
    net.cpu()
    obelisk.cpu()
    torch.save(net.state_dict(), 'midl_augment_net_obelisk'+str(fold)+'.pth')
    torch.save(obelisk.state_dict(), 'midl_augment_offset_obelisk'+str(fold)+'.pth')
    

sys.stdout = old_stdout
log_file.close()

