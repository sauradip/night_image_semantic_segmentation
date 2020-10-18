# import numpy as np
#!/usr/bin/python
# -*- encoding: utf-8 -*-




class Config(object):
    def __init__(self):
        
        ## dataset
        self.data_dir = '/home/sauradip/Desktop/code/icip19/data'
        self.data_list = '/home/sauradip/Desktop/code/icip19/adapt/AdaptSegNet/dataset/cityscapes_list/val.txt'
        self.result_dir = '/home/sauradip/Desktop/code/icip19/results'
        self.split = 'val'
        self.n_classes = 19
        self.crop_size = 256
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.img_mean = (104.00698793,116.66876762,122.67891434)

        ## network
        self.stride = 16
        self.real_loss = 'cross_entropy'   
        self.backbone = 'deeplabv3plus_mobilenet'

        ## ckpt
        self.real_ckpt = '/home/sauradip/Desktop/code/icip19/real/DeepLabV3Plus-Pytorch/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
        self.adapt_ckpt = '/home/sauradip/Desktop/code/icip19/adapt/AdaptSegNet/model/GTA2Cityscapes_vgg-ac4ac9f6.pth'
        self.backbone_ckpt = '/home/sauradip/Desktop/code/icip19/real/DeepLabV3Plus-Pytorch/ckpt/mobilenet_v2-b0353104.pth'                                                                       
        self.model_ckpt = 'checkpoint_real/'

        ## model and loss
        self.ignore_label = 255
        self.aspp_global_feature = False

        ## train control
        self.cur_epochs = 0
        self.cur_itrs = 0
        self.total_iter = 30000
        self.batch = 2 #### keep batch > 1 otherwise code fails
        self.weight_decay = 1e-4
        self.lr_policy = 'poly'
        self.l_r = 0.01
        self.step_size = 10000
        self.momentum = 0.01
       
