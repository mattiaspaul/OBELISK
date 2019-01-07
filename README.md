# OBELISK one binary extremely large and inflecting sparse kernel 
(pytorch v1.0 implementation) 

This repository contains code for the Medical Image Anaylsis (MIDL Special Issue) paper:
OBELISK-Net: Fewer Layers to Solve 3D Multi-Organ Segmentation with Sparse Deformable Convolutions
by Mattias P. Heinrich, Ozan Oktay, Nassim Bouteldja 
(winner of the MIDL 2018 best paper award)

The main idea of OBELISK is to learn a large spatially deformable filter kernel for (3D) image analysis. It replaces a conventional (say 5x5) convolution with 1) trainable spatial filter offsets xy(z)-coordinates and 2) a linear 1x1 convolution that contains the filter coefficients (values). During training OBELISK will adapt its receptive field to the given problem in a completely data-driven manner and thus automatically solve many tuning steps that are usually done by 'network engineering'. The OBELISK layers have substantially fewer trainable parameters than conventional CNNs used in 3D U-Nets and perform often better for medical segmentation tasks.

The working principle (and the basis of its implementation) are visualised below:

![overview](obelisk_explanation_slide_github.png)

please see details in the upcoming MEDIA paper
or for now the original MIDL version: https://openreview.net/forum?id=BkZu9wooz

How to use this code:
