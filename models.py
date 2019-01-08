import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


#Hybrid OBELISK model trained using pytorch v0.4.1 with TCIA data
#model contains two obelisk layers combined with traditional CNNs
#the layers have 512 and 128 trainable offsets and 230k trainable weights in total
class obeliskhybrid_tcia(nn.Module):
    def __init__(self,num_labels):
        super(obeliskhybrid_tcia, self).__init__()
        self.num_labels = num_labels
        D_in5 = 18; H_in5 = 18; W_in5 = 18;
        D_in4 = 36; H_in4 = 36; W_in4 = 36;
        D_in3 = 72; H_in3 = 72; W_in3 = 72;
        D_in2 = 144; H_in2 = 144; W_in2 = 144;
        D_grid = D_in3; H_grid = H_in3; W_grid = W_in3;

        D_grid1 = D_in3; H_grid1 = H_in3; W_grid1 = W_in3;
        D_grid2 = D_in4; H_grid2 = H_in4; W_grid2 = W_in4;
        D_grid3 = D_in5; H_grid3 = H_in4; W_grid3 = W_in5;

        self.D_grid1 = D_grid1; self.H_grid1 = H_grid1; self.W_grid1 = W_grid1;
        self.D_grid2 = D_grid2; self.H_grid2 = H_grid2; self.W_grid2 = W_grid2;
        self.D_grid3 = D_grid3; self.H_grid3 = H_grid3; self.W_grid3 = W_grid3;
        #U-Net Encoder
        self.conv0 = nn.Conv3d(1, 4, 3, padding=1)
        self.batch0 = nn.BatchNorm3d(4)
        self.conv1 = nn.Conv3d(4, 16, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm3d(16)
        self.conv11 = nn.Conv3d(16, 16, 3, padding=1)
        self.batch11 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(32)
        self.conv22 = nn.Conv3d(32, 32, 3, padding=1)
        self.batch22 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 32, 3, padding=1)
        self.batch3 = nn.BatchNorm3d(32)
        
        #Obelisk Encoder
        # sample_grid: 1 x    1     x #samples x 1 x 3
        # offsets:     1 x #offsets x     1    x 1 x 3
        self.sample_grid1 = torch.cat((torch.linspace(-1,1,W_grid2).view(1,1,-1,1).repeat(D_grid2,H_grid2,1,1),\
        torch.linspace(-1,1,H_grid2).view(1,-1,1,1).repeat(D_grid2,1,W_grid2,1),\
                                       torch.linspace(-1,1,D_grid2).view(-1,1,1,1).repeat(1,H_grid2,W_grid2,1)),dim=3).view(1,1,-1,1,3).detach()
        self.sample_grid1.requires_grad = False
        self.sample_grid2 = torch.cat((torch.linspace(-1,1,W_grid3).view(1,1,-1,1).repeat(D_grid3,H_grid3,1,1),\
        torch.linspace(-1,1,H_grid3).view(1,-1,1,1).repeat(D_grid3,1,W_grid3,1),\
        torch.linspace(-1,1,D_grid3).view(-1,1,1,1).repeat(1,H_grid3,W_grid3,1)),dim=3).view(1,1,-1,1,3).detach()
        self.sample_grid2.requires_grad = False
        self.offset1 = nn.Parameter(torch.randn(1,128,1,1,3)*0.05)
        self.linear1a = nn.Conv3d(128,128,1,groups=2,bias=False)
        self.batch1a = nn.BatchNorm3d(128)
        self.linear1b = nn.Conv3d(128,32,1,bias=False)
        self.batch1b = nn.BatchNorm3d(128+32)
        self.linear1c = nn.Conv3d(128+32,32,1,bias=False)
        self.batch1c = nn.BatchNorm3d(128+64)
        self.linear1d = nn.Conv3d(128+64,32,1,bias=False)
        self.batch1d = nn.BatchNorm3d(128+96)
        self.linear1e = nn.Conv3d(128+96,16,1,bias=False)
        
        self.offset2 = nn.Parameter(torch.randn(1,512,1,1,3)*0.05)
        self.linear2a = nn.Conv3d(512,128,1,groups=4,bias=False)
        self.batch2a = nn.BatchNorm3d(128)
        self.linear2b = nn.Conv3d(128,32,1,bias=False)
        self.batch2b = nn.BatchNorm3d(128+32)
        self.linear2c = nn.Conv3d(128+32,32,1,bias=False)
        self.batch2c = nn.BatchNorm3d(128+64)
        self.linear2d = nn.Conv3d(128+64,32,1,bias=False)
        self.batch2d = nn.BatchNorm3d(128+96)
        self.linear2e = nn.Conv3d(128+96,32,1,bias=False)
        
    

        
        #U-Net Decoder 
        self.conv6bU = nn.Conv3d(64, 32, 3, padding=1)#96#64#32
        self.batch6bU = nn.BatchNorm3d(32)
        self.conv6U = nn.Conv3d(64, 12, 3, padding=1)#64#48
        self.batch6U = nn.BatchNorm3d(12)
        self.conv7U = nn.Conv3d(16, num_labels, 3, padding=1)#24#16#24
        self.batch7U = nn.BatchNorm3d(num_labels)
        self.conv77U = nn.Conv3d(num_labels, num_labels, 1)
        
    def forward(self, inputImg):
    
        B,C,D,H,W = inputImg.size()
        device = inputImg.device

        leakage = 0.025
        x0 = F.avg_pool3d(inputImg,3,padding=1,stride=1)
        x00 = F.avg_pool3d(F.avg_pool3d(inputImg,3,padding=1,stride=1),3,padding=1,stride=1)
        
        x1 = F.leaky_relu(self.batch0(self.conv0(inputImg)), leakage)
        x = F.leaky_relu(self.batch1(self.conv1(x1)),leakage)
        x2 = F.leaky_relu(self.batch11(self.conv11(x)),leakage)
        x = F.leaky_relu(self.batch2(self.conv2(x2)),leakage)
        x = F.leaky_relu(self.batch22(self.conv22(x)),leakage)

        x = F.leaky_relu(self.batch3(self.conv3(x)),leakage)

        x_o1 = F.grid_sample(x0, (self.sample_grid1.to(device).repeat(B,1,1,1,1) + self.offset1)).view(B,-1,self.D_grid2,self.H_grid2,self.W_grid2)
        x_o1 = F.leaky_relu(self.linear1a(x_o1),leakage)
        x_o1a = torch.cat((x_o1,F.leaky_relu(self.linear1b(self.batch1a(x_o1)),leakage)),dim=1)
        x_o1b = torch.cat((x_o1a,F.leaky_relu(self.linear1c(self.batch1b(x_o1a)),leakage)),dim=1)
        x_o1c = torch.cat((x_o1b,F.leaky_relu(self.linear1d(self.batch1c(x_o1b)),leakage)),dim=1)
        x_o1d = F.leaky_relu(self.linear1e(self.batch1d(x_o1c)),leakage)
        x_o1 = F.interpolate(x_o1d, size=[self.D_grid1,self.H_grid1,self.W_grid1], mode='trilinear', align_corners=False)

        x_o2 = F.grid_sample(x00, (self.sample_grid2.to(device).repeat(B,1,1,1,1) + self.offset2)).view(B,-1,self.D_grid3,self.H_grid3,self.W_grid3)
        x_o2 = F.leaky_relu(self.linear2a(x_o2),leakage)
        x_o2a = torch.cat((x_o2,F.leaky_relu(self.linear2b(self.batch2a(x_o2)),leakage)),dim=1)
        x_o2b = torch.cat((x_o2a,F.leaky_relu(self.linear2c(self.batch2b(x_o2a)),leakage)),dim=1)
        x_o2c = torch.cat((x_o2b,F.leaky_relu(self.linear2d(self.batch2c(x_o2b)),leakage)),dim=1)
        x_o2d = F.leaky_relu(self.linear2e(self.batch2d(x_o2c)),leakage)
        x_o2 = F.interpolate(x_o2d, size=[self.D_grid2,self.H_grid2,self.W_grid2], mode='trilinear', align_corners=False)

        x = F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x,x_o2),1))),leakage)
        x = F.interpolate(x, size=[self.D_grid1,self.H_grid1,self.W_grid1], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x,x_o1,x2),1))),leakage)
        x = F.interpolate(x, size=[D,H,W], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch7U(self.conv7U(torch.cat((x,x1),1))),leakage)
       
        x = self.conv77U(x)
        
        return x
    
#Hybrid OBELISK CNN model that contains two obelisk layers combined with traditional CNNs
#the layers have 512 and 128 trainable offsets and 230k trainable weights in total
#trained with pytorch v1.0 for VISCERAL data
class obeliskhybrid_visceral(nn.Module):
    def __init__(self,num_labels,full_res):
        super(obeliskhybrid_visceral, self).__init__()
        self.num_labels = num_labels
        D_in1 = full_res[0]; H_in1 = full_res[1]; W_in1 = full_res[2];
        D_in2 = (D_in1+1)//2; H_in2 = (H_in1+1)//2; W_in2 = (W_in1+1)//2; #half resolution
        self.half_res = torch.Tensor([D_in2,H_in2,W_in2]).long(); half_res = self.half_res
        D_in4 = (D_in2+1)//2; H_in4 = (H_in2+1)//2; W_in4 = (W_in2+1)//2; #quarter resolution
        self.quarter_res = torch.Tensor([D_in4,H_in4,W_in4]).long(); quarter_res = self.quarter_res
        D_in8 = (D_in4+1)//2; H_in8 = (H_in4+1)//2; W_in8 = (W_in4+1)//2; #eighth resolution
        self.eighth_res = torch.Tensor([D_in8,H_in8,W_in8]).long(); eighth_res = self.eighth_res

        #U-Net Encoder
        self.conv0 = nn.Conv3d(1, 4, 3, padding=1)
        self.batch0 = nn.BatchNorm3d(4)
        self.conv1 = nn.Conv3d(4, 16, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm3d(16)
        self.conv11 = nn.Conv3d(16, 16, 3, padding=1)
        self.batch11 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(32)
        
        # Obelisk Encoder (for simplicity using regular sampling grid)
        # the first obelisk layer has 128 the second 512 trainable offsets
        # sample_grid: 1 x    1     x #samples x 1 x 3
        # offsets:     1 x #offsets x     1    x 1 x 3
        self.sample_grid1 = F.affine_grid(torch.eye(3,4).unsqueeze(0),torch.Size((1,1,quarter_res[0],quarter_res[1],quarter_res[2]))).view(1,1,-1,1,3).detach()
        self.sample_grid1.requires_grad = False
        self.sample_grid2 = F.affine_grid(torch.eye(3,4).unsqueeze(0),torch.Size((1,1,eighth_res[0],eighth_res[1],eighth_res[2]))).view(1,1,-1,1,3).detach()
        self.sample_grid2.requires_grad = False
        
        self.offset1 = nn.Parameter(torch.randn(1,128,1,1,3)*0.05)
        self.linear1a = nn.Conv3d(4*128,128,1,groups=4,bias=False)
        self.batch1a = nn.BatchNorm3d(128)
        self.linear1b = nn.Conv3d(128,32,1,bias=False)
        self.batch1b = nn.BatchNorm3d(128+32)
        self.linear1c = nn.Conv3d(128+32,32,1,bias=False)
        self.batch1c = nn.BatchNorm3d(128+64)
        self.linear1d = nn.Conv3d(128+64,32,1,bias=False)
        self.batch1d = nn.BatchNorm3d(128+96)
        self.linear1e = nn.Conv3d(128+96,32,1,bias=False)
        
        self.offset2 = nn.Parameter(torch.randn(1,512,1,1,3)*0.05)
        self.linear2a = nn.Conv3d(512,128,1,groups=4,bias=False)
        self.batch2a = nn.BatchNorm3d(128)
        self.linear2b = nn.Conv3d(128,32,1,bias=False)
        self.batch2b = nn.BatchNorm3d(128+32)
        self.linear2c = nn.Conv3d(128+32,32,1,bias=False)
        self.batch2c = nn.BatchNorm3d(128+64)
        self.linear2d = nn.Conv3d(128+64,32,1,bias=False)
        self.batch2d = nn.BatchNorm3d(128+96)
        self.linear2e = nn.Conv3d(128+96,32,1,bias=False)
        
        #U-Net Decoder 
        self.conv6bU = nn.Conv3d(64, 32, 3, padding=1)
        self.batch6bU = nn.BatchNorm3d(32)
        self.conv6U = nn.Conv3d(64+16, 32, 3, padding=1)
        self.batch6U = nn.BatchNorm3d(32)
        self.conv8 = nn.Conv3d(32, num_labels, 1)
        
    def forward(self, inputImg):
    
        B,C,D,H,W = inputImg.size()
        device = inputImg.device
        leakage = 0.05 #leaky ReLU used for conventional CNNs
        
        #unet-encoder
        x00 = F.avg_pool3d(inputImg,3,padding=1,stride=1)
        
        x1 = F.leaky_relu(self.batch0(self.conv0(inputImg)), leakage)
        x = F.leaky_relu(self.batch1(self.conv1(x1)),leakage)
        x2 = F.leaky_relu(self.batch11(self.conv11(x)),leakage)
        x = F.leaky_relu(self.batch2(self.conv2(x2)),leakage)
        
        #in this model two obelisk layers with fewer spatial offsets are used
        #obelisk layer 1
        x_o1 = F.grid_sample(x1, (self.sample_grid1.to(device).repeat(B,1,1,1,1) + self.offset1)).view(B,-1,self.quarter_res[0],self.quarter_res[1],self.quarter_res[2])
        #1x1 kernel dense-net
        x_o1 = F.relu(self.linear1a(x_o1))
        x_o1a = torch.cat((x_o1,F.relu(self.linear1b(self.batch1a(x_o1)))),dim=1)
        x_o1b = torch.cat((x_o1a,F.relu(self.linear1c(self.batch1b(x_o1a)))),dim=1)
        x_o1c = torch.cat((x_o1b,F.relu(self.linear1d(self.batch1c(x_o1b)))),dim=1)
        x_o1d = F.relu(self.linear1e(self.batch1d(x_o1c)))
        x_o1 = F.interpolate(x_o1d, size=[self.half_res[0],self.half_res[1],self.half_res[2]], mode='trilinear', align_corners=False)
        
        #obelisk layer 2
        x_o2 = F.grid_sample(x00, (self.sample_grid2.to(device).repeat(B,1,1,1,1) + self.offset2)).view(B,-1,self.eighth_res[0],self.eighth_res[1],self.eighth_res[2])
        x_o2 = F.relu(self.linear2a(x_o2))
        #1x1 kernel dense-net
        x_o2a = torch.cat((x_o2,F.relu(self.linear2b(self.batch2a(x_o2)))),dim=1)
        x_o2b = torch.cat((x_o2a,F.relu(self.linear2c(self.batch2b(x_o2a)))),dim=1)
        x_o2c = torch.cat((x_o2b,F.relu(self.linear2d(self.batch2c(x_o2b)))),dim=1)
        x_o2d = F.relu(self.linear2e(self.batch2d(x_o2c)))
        x_o2 = F.interpolate(x_o2d, size=[self.quarter_res[0],self.quarter_res[1],self.quarter_res[2]], mode='trilinear', align_corners=False)

        #unet-decoder
        x = F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x,x_o2),1))),leakage)
        x = F.interpolate(x, size=[self.half_res[0],self.half_res[1],self.half_res[2]], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x,x_o1,x2),1))),leakage)
        x = F.interpolate(self.conv8(x), size=[D,H,W], mode='trilinear', align_corners=False)
        
        return x

#original OBELISK model as described in MIDL2018 paper
#contains around 130k trainable parameters and 1024 binary offsets
#most simple Obelisk-Net with one deformable convolution followed by 1x1 Dense-Net
class obelisk_visceral(nn.Module):
    def __init__(self,num_labels,full_res):
        super(obelisk_visceral, self).__init__()
        self.num_labels = num_labels
        self.full_res = full_res
        D_in1 = full_res[0]; H_in1 = full_res[1]; W_in1 = full_res[2];
        D_in2 = (D_in1+1)//2; H_in2 = (H_in1+1)//2; W_in2 = (W_in1+1)//2; #half resolution
        self.half_res = torch.Tensor([D_in2,H_in2,W_in2]).long(); half_res = self.half_res
        D_in4 = (D_in2+1)//2; H_in4 = (H_in2+1)//2; W_in4 = (W_in2+1)//2; #quarter resolution
        self.quarter_res = torch.Tensor([D_in4,H_in4,W_in4]).long(); quarter_res = self.quarter_res
        
        #Obelisk Layer
        # sample_grid: 1 x    1     x #samples x 1 x 3
        # offsets:     1 x #offsets x     1    x 1 x 3
        
        self.sample_grid1 = F.affine_grid(torch.eye(3,4).unsqueeze(0),torch.Size((1,1,quarter_res[0],quarter_res[1],quarter_res[2])))
        self.sample_grid1.requires_grad = False
        
        #in this model (binary-variant) two spatial offsets are paired 
        self.offset1 = nn.Parameter(torch.randn(1,1024,1,2,3)*0.05)
        
        #Dense-Net with 1x1x1 kernels
        self.LIN1 = nn.Conv3d(1024, 256, 1, bias=False, groups=4) #grouped convolutions
        self.BN1 = nn.BatchNorm3d(256)
        self.LIN2 = nn.Conv3d(256, 128, 1, bias=False)
        self.BN2 = nn.BatchNorm3d(128)
        
        self.LIN3a = nn.Conv3d(128, 32, 1,bias=False)
        self.BN3a = nn.BatchNorm3d(128+32)
        self.LIN3b = nn.Conv3d(128+32, 32, 1,bias=False)
        self.BN3b = nn.BatchNorm3d(128+64)
        self.LIN3c = nn.Conv3d(128+64, 32, 1,bias=False)
        self.BN3c = nn.BatchNorm3d(128+96)
        self.LIN3d = nn.Conv3d(128+96, 32, 1,bias=False)
        self.BN3d = nn.BatchNorm3d(256)
        
        self.LIN4 = nn.Conv3d(256, num_labels,1)

        
    def forward(self, inputImg, sample_grid=None):
    
        B,C,D,H,W = inputImg.size()
        if(sample_grid is None):
            sample_grid = self.sample_grid1
        sample_grid = sample_grid.to(inputImg.device)    
        #pre-smooth image (has to be done in advance for original models )
        #x00 = F.avg_pool3d(inputImg,3,padding=1,stride=1)
        
        _,D_grid,H_grid,W_grid,_ = sample_grid.size()
        input = F.grid_sample(inputImg, (sample_grid.view(1,1,-1,1,3).repeat(B,1,1,1,1) + self.offset1[:,:,:,0:1,:])).view(B,-1,D_grid,H_grid,W_grid)-\
        F.grid_sample(inputImg, (sample_grid.view(1,1,-1,1,3).repeat(B,1,1,1,1) + self.offset1[:,:,:,1:2,:])).view(B,-1,D_grid,H_grid,W_grid)
        
        x1 = F.relu(self.BN1(self.LIN1(input)))
        x2 = self.BN2(self.LIN2(x1))
        
        x3a = torch.cat((x2,F.relu(self.LIN3a(x2))),dim=1)
        x3b = torch.cat((x3a,F.relu(self.LIN3b(self.BN3a(x3a)))),dim=1)
        x3c = torch.cat((x3b,F.relu(self.LIN3c(self.BN3b(x3b)))),dim=1)
        x3d = torch.cat((x3c,F.relu(self.LIN3d(self.BN3c(x3c)))),dim=1)

        x4 = self.LIN4(self.BN3d(x3d))
        #return half-resolution segmentation/prediction 
        return F.interpolate(x4, size=[self.half_res[0],self.half_res[1],self.half_res[2]], mode='trilinear',align_corners=False)



class globalfcnet_tcia(nn.Module):

    def __init__(self, num_labels):

        super(globalfcnet_tcia, self).__init__()

        # # 1) parameterintensive Unet with 2 global FC-Layer [16mio parameters]

        self.conv0 = nn.Conv3d(1, 5, 3, padding=1)
        self.batch0 = nn.BatchNorm3d(5)

        self.conv1 = nn.Conv3d(5, 10, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm3d(10)
        self.conv11 = nn.Conv3d(10, 10, 3, padding=1)
        self.batch11 = nn.BatchNorm3d(10)

        self.conv2 = nn.Conv3d(10, 30, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(30)
        self.conv22 = nn.Conv3d(30, 30, 3, padding=1)
        self.batch22 = nn.BatchNorm3d(30)

        self.conv3 = nn.Conv3d(30, 50, 3, stride=2, padding=1)
        self.batch3 = nn.BatchNorm3d(50)
        self.conv33 = nn.Conv3d(50, 50, 3, padding=1)
        self.batch33 = nn.BatchNorm3d(50)

        self.conv4 = nn.Conv3d(50, 20, 3, stride=2, padding=1)
        self.batch4 = nn.BatchNorm3d(20)
        self.conv44 = nn.Conv3d(20, 20, 3, padding=1)
        self.batch44 = nn.BatchNorm3d(20)

        self.fc1 = nn.Linear(20 * 9 * 9 * 9, 400)

        self.fc2U = nn.Linear(400, 20 * 9 * 9 * 9)

        self.conv6dU = nn.Conv3d(40, 20, 3, padding=1)
        self.batch6dU = nn.BatchNorm3d(20)

        self.conv6cU = nn.Conv3d(70, 20, 3, padding=1)
        self.batch6cU = nn.BatchNorm3d(20)

        self.conv6bU = nn.Conv3d(50, 10, 3, padding=1)
        self.batch6bU = nn.BatchNorm3d(10)

        self.conv6U = nn.Conv3d(20, 8, 3, padding=1)
        self.batch6U = nn.BatchNorm3d(8)

        self.conv7U = nn.Conv3d(13, num_labels, 3, padding=1)
        self.batch7U = nn.BatchNorm3d(num_labels)
        self.conv77U = nn.Conv3d(num_labels, num_labels, 3, padding=1)
  
    def forward(self, inputImg):

        # 1) forward for global Unet [16mio parameters]
        x1 = F.leaky_relu(self.batch0(self.conv0(inputImg)), 0.1)

        x = F.leaky_relu(self.batch1(self.conv1(x1)),0.1)
        x2 = F.leaky_relu(self.batch11(self.conv11(x)),0.1)

        x = F.leaky_relu(self.batch2(self.conv2(x2)),0.1)
        x3 = F.leaky_relu(self.batch22(self.conv22(x)),0.1)

        x = F.leaky_relu(self.batch3(self.conv3(x3)),0.1)
        x4 = F.leaky_relu(self.batch33(self.conv33(x)),0.1)

        x = F.leaky_relu(self.batch4(self.conv4(x4)),0.1)
        x5 = F.leaky_relu(self.batch44(self.conv44(x)),0.1)

        sizeX = x5.size()
        x = x5.view(-1, 20 * 9 * 9 * 9)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)

        x = self.fc2U(x)
        x = F.dropout(x, training=self.training)
        x = x.view(sizeX)

        x = F.leaky_relu(self.batch6dU(self.conv6dU(torch.cat((x,x5),1))),0.1)

        x = F.interpolate(x, size=[18,18,18], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6cU(self.conv6cU(torch.cat((x,x4),1))),0.1)

        x = F.interpolate(x, size=[36,36,36], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x,x3),1))),0.1)

        x = F.interpolate(x, size=[72,72,72], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x,x2),1))),0.1)

        x = F.interpolate(x, size=[144,144,144], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch7U(self.conv7U(torch.cat((x,x1),1))),0.1)
        # x = F.sigmoid(self.conv77U(x))
        x = self.conv77U(x)
        
        return x


class allconvunet_tcia(nn.Module):

    def __init__(self, num_labels):

        super(allconvunet_tcia, self).__init__()


        # # 4) Deeper classic all-conv.-Unet [880k parameters]
        self.conv0 = nn.Conv3d(1, 5, 3, padding=1)
        self.batch0 = nn.BatchNorm3d(5)

        self.conv1 = nn.Conv3d(5, 14, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm3d(14)
        self.conv11 = nn.Conv3d(14, 14, 3, padding=1)
        self.batch11 = nn.BatchNorm3d(14)

        self.conv2 = nn.Conv3d(14, 28, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(28)
        self.conv22 = nn.Conv3d(28, 28, 3, padding=1)
        self.batch22 = nn.BatchNorm3d(28)

        self.conv3 = nn.Conv3d(28, 42, 3, stride=2, padding=1)
        self.batch3 = nn.BatchNorm3d(42)
        self.conv33 = nn.Conv3d(42, 42, 3, padding=1)
        self.batch33 = nn.BatchNorm3d(42)

        self.conv4 = nn.Conv3d(42, 56, 3, stride=2, padding=1)
        self.batch4 = nn.BatchNorm3d(56)
        self.conv44 = nn.Conv3d(56, 56, 3, padding=1)
        self.batch44 = nn.BatchNorm3d(56)

        self.conv5 = nn.Conv3d(56, 70, 3, stride=2, padding=1)
        self.batch5 = nn.BatchNorm3d(70)
        self.conv55 = nn.Conv3d(70, 70, 3, padding=1)
        self.batch55 = nn.BatchNorm3d(70)

        self.conv6dU = nn.Conv3d(126, 56, 3, padding=1)
        self.batch6dU = nn.BatchNorm3d(56)

        self.conv6cU = nn.Conv3d(98, 42, 3, padding=1)
        self.batch6cU = nn.BatchNorm3d(42)

        self.conv6bU = nn.Conv3d(70, 28, 3, padding=1)
        self.batch6bU = nn.BatchNorm3d(28)

        self.conv6U = nn.Conv3d(42, 14, 3, padding=1)
        self.batch6U = nn.BatchNorm3d(14)

        self.conv7U = nn.Conv3d(19, num_labels, 3, padding=1)
        self.batch7U = nn.BatchNorm3d(num_labels)
        self.conv77U = nn.Conv3d(num_labels, num_labels, 3, padding=1)
    

    def forward(self, inputImg):

        # # 3) forward for classical deeper all-conv.-Unet [880k parameters] + dropout
        x1 = F.dropout3d(F.leaky_relu(self.batch0(self.conv0(inputImg)), 0.1))

        x = F.dropout3d(F.leaky_relu(self.batch1(self.conv1(x1)),0.1))
        x2 = F.dropout3d(F.leaky_relu(self.batch11(self.conv11(x)),0.1))

        x = F.dropout3d(F.leaky_relu(self.batch2(self.conv2(x2)),0.1))
        x3 = F.dropout3d(F.leaky_relu(self.batch22(self.conv22(x)),0.1))

        x = F.dropout3d(F.leaky_relu(self.batch3(self.conv3(x3)),0.1))
        x4 = F.dropout3d(F.leaky_relu(self.batch33(self.conv33(x)),0.1))

        x = F.dropout3d(F.leaky_relu(self.batch4(self.conv4(x4)),0.1))
        x5 = F.dropout3d(F.leaky_relu(self.batch44(self.conv44(x)),0.1))

        x = F.dropout3d(F.leaky_relu(self.batch5(self.conv5(x5)),0.1))
        x = F.dropout3d(F.leaky_relu(self.batch55(self.conv55(x)),0.1))

        x = F.interpolate(x, size=[9,9,9], mode='trilinear', align_corners=False)
        x = F.dropout3d(F.leaky_relu(self.batch6dU(self.conv6dU(torch.cat((x,x5),1))),0.1))

        x = F.interpolate(x, size=[18,18,18], mode='trilinear', align_corners=False)
        x = F.dropout3d(F.leaky_relu(self.batch6cU(self.conv6cU(torch.cat((x,x4),1))),0.1))

        x = F.interpolate(x, size=[36,36,36], mode='trilinear', align_corners=False)
        x = F.dropout3d(F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x,x3),1))),0.1))

        x = F.interpolate(x, size=[72,72,72], mode='trilinear', align_corners=False)
        x = F.dropout3d(F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x,x2),1))),0.1))

        x = F.interpolate(x, size=[144,144,144], mode='trilinear', align_corners=False)
        x = F.dropout3d(F.leaky_relu(self.batch7U(self.conv7U(torch.cat((x,x1),1))),0.1))

        x = self.conv77U(x)

        return x

class globalfcnet_visceral(nn.Module):

    def __init__(self):

        super(globalfcnet_visceral, self).__init__()

        self.conv0 = nn.Conv3d(1, 5, 3, padding=1)
        self.batch0 = nn.BatchNorm3d(5)

        self.conv1 = nn.Conv3d(5, 10, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm3d(10)
        self.conv11 = nn.Conv3d(10, 10, 3, padding=1)
        self.batch11 = nn.BatchNorm3d(10)

        self.conv2 = nn.Conv3d(10, 30, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(30)
        self.conv22 = nn.Conv3d(30, 30, 3, padding=1)
        self.batch22 = nn.BatchNorm3d(30)

        self.conv3 = nn.Conv3d(30, 50, 3, stride=2, padding=1)
        self.batch3 = nn.BatchNorm3d(50)
        self.conv33 = nn.Conv3d(50, 50, 3, padding=1)
        self.batch33 = nn.BatchNorm3d(50)

        self.conv4 = nn.Conv3d(50, 20, 3, stride=2, padding=1)
        self.batch4 = nn.BatchNorm3d(20)
        self.conv44 = nn.Conv3d(20, 20, 3, padding=1)
        self.batch44 = nn.BatchNorm3d(20)

        self.fc1 = nn.Linear(20 * 10 * 8 * 10, 400)

        self.fc2U = nn.Linear(400, 20 * 10 * 8 * 10)

        self.conv6dU = nn.Conv3d(40, 20, 3, padding=1)
        self.batch6dU = nn.BatchNorm3d(20)

        self.conv6cU = nn.Conv3d(70, 20, 3, padding=1)
        self.batch6cU = nn.BatchNorm3d(20)

        self.conv6bU = nn.Conv3d(50, 10, 3, padding=1)
        self.batch6bU = nn.BatchNorm3d(10)

        self.conv6U = nn.Conv3d(20, 8, 3, padding=1)
        self.batch6U = nn.BatchNorm3d(8)

        self.conv7U = nn.Conv3d(13, 8, 3, padding=1)
        self.batch7U = nn.BatchNorm3d(8)
        self.conv77U = nn.Conv3d(8, 8, 3, padding=1)




    def forward(self, inputImg):

        x1 = F.leaky_relu(self.batch0(self.conv0(inputImg)), 0.1)

        x = F.leaky_relu(self.batch1(self.conv1(x1)),0.1)
        x2 = F.leaky_relu(self.batch11(self.conv11(x)),0.1)

        x = F.leaky_relu(self.batch2(self.conv2(x2)),0.1)
        x3 = F.leaky_relu(self.batch22(self.conv22(x)),0.1)

        x = F.leaky_relu(self.batch3(self.conv3(x3)),0.1)
        x4 = F.leaky_relu(self.batch33(self.conv33(x)),0.1)

        x = F.leaky_relu(self.batch4(self.conv4(x4)),0.1)
        x5 = F.leaky_relu(self.batch44(self.conv44(x)),0.1)

        sizeX = x5.size()
        x = x5.view(-1, 20 * 10 * 8 * 10)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)

        xU = self.fc2U(x)
        xU = F.dropout(xU, training=self.training)
        xU = xU.view(sizeX)

        xU = F.leaky_relu(self.batch6dU(self.conv6dU(torch.cat((xU,x5),1))),0.1)

        xU = F.interpolate(xU, size=[20,15,20], mode='trilinear', align_corners=False)
        xU = F.leaky_relu(self.batch6cU(self.conv6cU(torch.cat((xU,x4),1))),0.1)

        xU = F.interpolate(xU, size=[39,29,40], mode='trilinear', align_corners=False)
        xU = F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((xU,x3),1))),0.1)

        xU = F.interpolate(xU, size=[78,58,80], mode='trilinear', align_corners=False)
        xU = F.leaky_relu(self.batch6U(self.conv6U(torch.cat((xU,x2),1))),0.1)

        xU = F.interpolate(xU, size=[156,115,160], mode='trilinear', align_corners=False)
        xU = F.leaky_relu(self.batch7U(self.conv7U(torch.cat((xU,x1),1))),0.1)
        # xU = F.sigmoid(self.conv77U(xU))
        xU = self.conv77U(xU)

        return xU
    
class allconvunet_visceral(nn.Module):

    def __init__(self):

        super(allconvunet_visceral, self).__init__()

        self.conv0 = nn.Conv3d(1, 5, 3, padding=1)
        self.batch0 = nn.BatchNorm3d(5)

        self.conv1 = nn.Conv3d(5, 16, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm3d(16)
        self.conv11 = nn.Conv3d(16, 16, 3, padding=1)
        self.batch11 = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(32)
        self.conv22 = nn.Conv3d(32, 32, 3, padding=1)
        self.batch22 = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.batch3 = nn.BatchNorm3d(64)
        self.conv33 = nn.Conv3d(64, 64, 3, padding=1)
        self.batch33 = nn.BatchNorm3d(64)

        self.conv4 = nn.Conv3d(64, 80, 3, stride=2, padding=1)
        self.batch4 = nn.BatchNorm3d(80)
        self.conv44 = nn.Conv3d(80, 80, 3, padding=1)
        self.batch44 = nn.BatchNorm3d(80)

        self.conv6cU = nn.Conv3d(144, 64, 3, padding=1)
        self.batch6cU = nn.BatchNorm3d(64)

        self.conv6bU = nn.Conv3d(96, 32, 3, padding=1)
        self.batch6bU = nn.BatchNorm3d(32)

        self.conv6U = nn.Conv3d(48, 8, 3, padding=1)
        self.batch6U = nn.BatchNorm3d(8)

        self.conv7U = nn.Conv3d(13, 8, 3, padding=1)
        self.batch7U = nn.BatchNorm3d(8)
        self.conv77U = nn.Conv3d(8, 8, 3, padding=1)


    def forward(self, inputImg):

        x1 = F.leaky_relu(self.batch0(self.conv0(inputImg)), 0.1)

        x = F.leaky_relu(self.batch1(self.conv1(x1)),0.1)
        x2 = F.leaky_relu(self.batch11(self.conv11(x)),0.1)

        x = F.leaky_relu(self.batch2(self.conv2(x2)),0.1)
        x3 = F.leaky_relu(self.batch22(self.conv22(x)),0.1)

        x = F.leaky_relu(self.batch3(self.conv3(x3)),0.1)
        x4 = F.leaky_relu(self.batch33(self.conv33(x)),0.1)

        x = F.leaky_relu(self.batch4(self.conv4(x4)),0.1)
        x = F.leaky_relu(self.batch44(self.conv44(x)),0.1)

        x = F.interpolate(x, size=[20,15,20], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6cU(self.conv6cU(torch.cat((x,x4),1))),0.1)

        x = F.interpolate(x, size=[39,29,40], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x,x3),1))),0.1)

        x = F.interpolate(x, size=[78,58,80], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x,x2),1))),0.1)

        x = F.interpolate(x, size=[156,115,160], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch7U(self.conv7U(torch.cat((x,x1),1))),0.1)
        # x = F.sigmoid(self.conv77U(xU))
        x = self.conv77U(x)

        return x
    
    
    