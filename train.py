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
import sys

import argparse

cuda_idx = 0

from utils import init_weights, countParam, augmentAffine, my_ohem, dice_coeff, Logger
from models import *#obeliskhybrid_tcia, obeliskhybrid_visceral
def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])

def main():
    #read/parse user command line input
    parser = argparse.ArgumentParser()

    parser.add_argument("-folder", dest="folder", help="training dataset folder", default='TCIA_CT', required=True)
    parser.add_argument("-scannumbers", dest="scannumbers", help="list of integers indicating which scans to use, \"1 2 3\" ", default=1, required=True, type=lambda s: [int(n) for n in s.split()])
    parser.add_argument("-filescan", dest="filescan", help="prototype scan filename i.e. pancreas_ct?.nii.gz", default='pancreas_ct?.nii.gz', required=True)
    parser.add_argument("-fileseg", dest="fileseg",  help="prototype segmentation name i.e. label_ct?.nii.gz", required=True)
    parser.add_argument("-output", dest="output",  help="filename (without extension) for output", default=None, required=True)
    #parser.add_argument("-groundtruth", dest="groundtruth",  help="nii.gz groundtruth segmentation", default=None, required=False)

    options = parser.parse_args()
    d_options = vars(options)
    #modelfilename = os.path.basename(d_options['model'])
    #modelname = split_at(modelfilename, '_', 1)[0]
    
    sys.stdout = Logger(d_options['output']+'_log.txt')

    # load train images and segmentations
    imgs = []
    segs = []
    scannumbers = d_options['scannumbers']
    print('scannumbers',scannumbers)
    if(d_options['filescan'].find("?")==-1):
        print('error filescan must contain \"?\" to insert numbers')
        exit()
    filesplit = split_at(d_options['filescan'],'?',1)
    filesegsplit = split_at(d_options['fileseg'],'?',1)

    for i in range(0, len(scannumbers)):
        #/share/data_rechenknecht01_1/heinrich/TCIA_CT
        filescan1 = filesplit[0]+str(scannumbers[i])+filesplit[1]
        img = nib.load(os.path.join(d_options['folder'],filescan1)).get_data()
        fileseg1 = filesegsplit[0]+str(scannumbers[i])+filesegsplit[1]
        seg = nib.load(os.path.join(d_options['folder'],fileseg1)).get_data()
        imgs.append(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float())
        segs.append(torch.from_numpy(seg).unsqueeze(0).long())

    imgs = torch.cat(imgs,0)
    segs = torch.cat(segs,0)
    imgs = imgs/1024.0 + 1.0 #scale data

    numEpoches = 300#1000
    batchSize = 4

    print('data loaded')

    class_weight = torch.sqrt(1.0/(torch.bincount(segs.view(-1)).float()))
    class_weight = class_weight/class_weight.mean()
    class_weight[0] = 0.5
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    print('inv sqrt class_weight',class_weight.data.cpu().numpy())

    num_labels = int(class_weight.numel())

    D_in1 = imgs.size(2); H_in1 = imgs.size(3); W_in1 = imgs.size(4); #full resolution
    full_res = torch.Tensor([D_in1,H_in1,W_in1]).long()


    net = obeliskhybrid_tcia(num_labels)
    net.apply(init_weights)
    print('obelisk params',countParam(net))

    print('initial offset std','%.3f'%(torch.std(net.offset1.data).item()))
    net.cuda(cuda_idx)

    #criterion = nn.CrossEntropyLoss()#
    my_criterion = my_ohem(.25,class_weight.cuda())#0.25 

    optimizer = optim.Adam(net.parameters(), lr=0.002, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.99)
    
    run_loss = np.zeros(300)

    dice_epoch = np.zeros((imgs.size(0),num_labels-1,300))
    fold_size = imgs.size(0)
    fold_size4 = fold_size - fold_size%4
    print('fold/batch sizes',fold_size,fold_size4,imgs.size(0))
    #for loop over iterations and epochs
    for epoch in range(300):


        net.train()

        run_loss[epoch] = 0.0
        t1 = 0.0

        idx_epoch = torch.randperm(fold_size)[:fold_size4].view(4,-1)
        t0 = time.time()

        for iter in range(idx_epoch.size(1)):
            idx = idx_epoch[:,iter]

            with torch.no_grad():
                imgs_cuda, y_label = augmentAffine(imgs[idx,:,:,:,:].cuda(), segs[idx,:,:,:].cuda(),strength=0.075)
                torch.cuda.empty_cache()

            optimizer.zero_grad() 
            
            #forward path and loss
            predict = net(imgs_cuda)

            loss = my_criterion(F.log_softmax(predict,dim=1),y_label)
            loss.backward()
            
            run_loss[epoch] += loss.item()
            optimizer.step()
            del loss; del predict; 
            torch.cuda.empty_cache()
            del imgs_cuda
            del y_label
            torch.cuda.empty_cache()
        scheduler.step()
        
        #evaluation on training images
        t1 = time.time()-t0
        net.eval()

        if(epoch%3==0):
            for testNo in range(imgs.size(0)):
                imgs_cuda = (imgs[testNo:testNo+1,:,:,:,:]).cuda()

                t0 = time.time() 
                predict = net(imgs_cuda)

                argmax = torch.max(predict,dim=1)[1]
                torch.cuda.synchronize()
                time_i = (time.time() - t0)
                dice_all = dice_coeff(argmax.cpu(), segs[testNo:testNo+1,:,:,:], num_labels)
                dice_epoch[testNo,:,epoch] = dice_all.numpy()
                #del output_test
                del predict
                del imgs_cuda
                torch.cuda.empty_cache()

            #print some feedback information
            print('epoch',epoch,'time train','%.3f'%t1,'time inf','%.3f'%time_i,'loss','%.3f'%(run_loss[epoch]),'stddev','%.3f'%(torch.std(net.offset1.data)))
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print('dice_avgs (training)',(np.nanmean(dice_epoch[:,:,epoch],0)*100.0))
            sys.stdout.saveCurrentResults()
            arr = {}
            arr['dice_epoch'] = dice_epoch#.numpy()

            scipy.io.savemat(d_options['output']+'.mat',arr)

        if(epoch%6==0):

            net.cpu()

            torch.save(net.state_dict(), d_options['output']+'.pth')

            net.cuda()

     
        


if __name__ == '__main__':
    main()


