import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from cityscapes_dataloader_v2 import cityscapesDataSet
from utils import ext_transforms as et
from adapt.AdaptSegNet.model.deeplab_vgg import DeeplabVGG
import network
import utils
from fusion import MultiScaleFusion
from sklearn.metrics import jaccard_score as jaccard_iou_loss
from config import config_factory


cfg = config_factory['cityscapes']



# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
# DATA_DIRECTORY = '/home/sauradip/Desktop/code/icip19/data'
# DATA_LIST_PATH = '/home/sauradip/Desktop/code/icip19/adapt/AdaptSegNet/dataset/cityscapes_list/val.txt'
# SAVE_PATH = '/home/sauradip/Desktop/code/icip19/results'
# SET = 'val'

IMG_MEAN = np.array(cfg.img_mean, dtype=np.float32)
DATA_DIRECTORY = cfg.data_dir
DATA_LIST_PATH = cfg.data_list
SAVE_PATH = cfg.result_dir
SET = cfg.split


class Nisenet(nn.Module):
    def __init__(self, real, adaptive, msfusion):
        super(Nisenet, self).__init__()

        self.real = real  ########## real channel #########
        self.adaptive = adaptive ######### adaptive channel #########
        ############# multi scale fusion ##########
        self.fusion = msfusion
        ##########################################
        self.interp = nn.Upsample(size=(256, 512), mode='bilinear', align_corners=True)

        # self.fusion = nn.Linear(4, 2)
        
    def forward(self, x1, x2):
        real_out = self.real(x1)
        real_copy = real_out
        adapt_out = self.adaptive(x2)
        adapt_copy = adapt_out
        real_out_2 = real_out.detach().max(dim=1)[1].cpu().numpy()
        real_out_1 = real_copy.detach().cpu().numpy()
        adapt_out_1 = interp(adapt_copy).detach().cpu().numpy()   
        adapt_out = interp(adapt_out).detach().max(dim=1)[1].cpu().numpy()
        # print("in net before",type(adapt_out))
        # adapt_out_1 = interp(adapt_out).detach().cpu().numpy()
        # distance = torch.dist(torch.from_numpy(real_out),torch.from_numpy(adapt_out),18)  
        # distance = F.pairwise_distance(torch.from_numpy(real_out).view(2,256*512).type(torch.FloatTensor),torch.from_numpy(adapt_out).view(2,256*512).type(torch.FloatTensor))
        # print("torch distance",distance)
        # targets = target_real.cpu().numpy()
        # x = torch.cat((x1, x2), dim=1)
        # x = self.fusion(F.relu(x))
        # print("adapt size", torch.from_numpy(adapt_out).size())
        # print("adapt size", torch.from_numpy(adapt_out).permute(0,2,3,1).size())
        # print()
        adapt_max = torch.from_numpy(adapt_out_1)
        out1,out2 = self.fusion(torch.from_numpy(real_out_2).unsqueeze(1).type('torch.FloatTensor'),torch.from_numpy(adapt_out).unsqueeze(1).type('torch.FloatTensor'))
        # print("in net after",type(adapt_out))
        # out1.requires_grad = True
        # out2.requires_grad = True
        # real_out_1.requires_grad = True
        # real_out_2.requires_grad = True
        # adapt_out.requires_grad = True
        adapt_max.requires_grad = True

        return out1 , out2 , real_out_1 , real_out_2, adapt_out , adapt_max



# model = Nisenet(real, adaptive)
# x1, x2 = torch.randn(1, 10), torch.randn(1, 20)
# output = model(x1, x2)



################### Model for Real Channel #####################

def real_channel():
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map['deeplabv3plus_mobilenet'](num_classes= cfg.n_classes, output_stride= cfg.stride)


    # if opts.separable_conv and 'plus' in opts.model:
    #     network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=cfg.momentum)
    
    # Set up metrics
    # metrics = StreamSegMetrics(opts.num_classes)
    l_r = cfg.l_r
    weight_decay = cfg.weight_decay
    lr_policy = cfg.lr_policy
    total_itrs = cfg.total_iter
    step_size = cfg.step_size
    loss_type = cfg.real_loss
    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*l_r},
        {'params': model.classifier.parameters(), 'lr': l_r},
    ], lr=l_r, momentum=0.9, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    return criterion, optimizer, scheduler, model

#######################################################################


####################### extra methods for training and loading ########

def save_ckpt(path):
    """ save current model
    """
    torch.save({
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
    }, path)
    print("Model saved as %s" % path)

def save_model_ckpt(path,model,cur_itrs):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
        }, path)
        print("Model saved as %s" % path)

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


class FusionLoss(torch.nn.Module):

      def __init__(self, margin=2.0):
            super(FusionLoss, self).__init__()
            self.margin = margin

      def forward(self, output1, output2, label):
            # Find the pairwise distance or eucledian distance of two output feature vectors
            euclidean_distance = F.pairwise_distance(output1, output2)
            # print("eucl dist", euclidean_distance)
            # print("eucl dist size", euclidean_distance.size())
            # euclidean_distance.requires_grad = True
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
            # loss_contrastive.requires_grad = True
            return loss_contrastive

class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JaccardLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

#####################################################################

################### Model for Adaptive Channel #####################

def adaptive_channel():
    model = DeeplabVGG(num_classes=19)
    # saved_state_dict = torch.load(args.restore_from,map_location='cpu')
    # model_dict = model.state_dict()
    # saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    # model_dict.update(saved_state_dict)

    return model

####################################################################


    # model.load_state_dict(saved_state_dict)



# Create models and load state_dicts   

print("Setting Up Real Channel")
real_loss, real_optim, real_schedule, real = real_channel()
print("\n Setting up Adaptive Channel")
adaptive = adaptive_channel()
print("\n Setting up Fusion Network")
fusion = MultiScaleFusion()

net = Nisenet(real,adaptive,fusion)
print("\n NiseNet is Created Successfully ")
print("\n ################################ \n")
optimizer_fuse = optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
# interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
interp = nn.Upsample(size=(cfg.crop_size, 2*cfg.crop_size), mode='bilinear', align_corners=True)

# Load state dicts
print("\n ###################### \n")
print("Loading Checkpoint Checkpoints ")

PATH_REAL = cfg.real_ckpt
PATH_ADAPT = cfg.adapt_ckpt

print("Loading Checkpoint of Real Channel")
real.load_state_dict(torch.load(PATH_REAL,map_location="cpu")["model_state"])
print("\n Successfully Loaded")
print("\n Loading Checkpoint of Adaptive Channel")
adaptive.load_state_dict(torch.load(PATH_ADAPT,map_location="cpu"))
print("\n Successfully Loaded")

device = torch.device("cpu")

print("\n What Device i am using ?", device)
real.to(device)
adaptive.to(device)
fusion.to(device)
net.to(device)





##### for real channel #########

# train_transform = et.ExtCompose([
#             #et.ExtResize( 512 ),
#             et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
#             et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
#             et.ExtRandomHorizontalFlip(),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])

val_transform = et.ExtCompose([
    et.ExtResize( cfg.crop_size ),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

# train_dst = Cityscapes(root=opts.data_root,
#                         split='train', transform=None)
# val_dst = Cityscapes(root=opts.data_root,
#                         split='val', transform=None)


########### for adapted channel ############

# targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
#                                                      max_iters=args.num_steps * args.iter_size * args.batch_size,
#                                                      crop_size=input_size_target,
#                                                      scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
#                                                      set=args.set),
#                                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
#                                    pin_memory=True)

testloader = data.DataLoader(cityscapesDataSet(DATA_DIRECTORY, DATA_LIST_PATH, crop_size=(512, 256), mean=IMG_MEAN, scale=False, mirror=False, set=SET, transform=val_transform),
                                    batch_size=cfg.batch, shuffle=False, pin_memory=True)

fusion_loss = FusionLoss()
jacc_loss = JaccardLoss()
mse_loss = torch.nn.MSELoss()
adapt_loss = utils.FocalLoss(ignore_index=255, size_average=True)



interval_loss = 0
cur_itrs = cfg.cur_itrs
cur_epochs = cfg.cur_epochs
total_iter = cfg.total_iter
output_stride = cfg.stride
############################# Training #########################
print(" \n ###################### Training ############### \n")
while True:
    net.train()
    cur_epochs += 1
    for index, batch in enumerate(testloader): 
        if index % 100 == 0:
            print('%d processd' % index)
        image_adapt, _, name,image_real,target_real= batch
        cur_itrs += 1

    ########### Both Shape : [batch, 3, 256, 512]) ##############

        # print("real shape", image_real.size())
        # print("adapt shape", image_adapt.size())

    #############################################################

        # output_adapt = adaptive(image_adapt)
        # output_adapt = interp(output_adapt).detach().cpu().numpy()
        # # output_adapt = output_adapt.transpose(1,2,0)

        image_real = image_real.to(device, dtype=torch.float32)
        target_real = target_real.to(device, dtype=torch.long)
        real_optim.zero_grad()
        optimizer_fuse.zero_grad()
        # output_real = real(image_real)
        # preds = output_real.detach().max(dim=1)[1].cpu().numpy() 
        # preds_real = output_real.detach().cpu().numpy()   
        # targets = target_real.cpu().numpy()
        
        ####### preds_real : real channel output ########
        ####### output_adapt : adaptive channel output #######

        # print("Adapt Shape", np.shape(output_adapt)) #### shape : (2,19,256,512,19)
        # print("Real Shape", np.shape(preds_real)) #### shape : (2,19,256,512)
        # print("Target Shape", np.shape(targets)) #### shape : (2,256,512)
        
        
        out1,out2,real_out, real_out_max, adapt_out , adapt_out_max = net(image_real,image_adapt)
        # print("+ve sample size", out1.size())
        # print("adapt size ", adapt_out_max.size())
        # print("after net real shape", torch.from_numpy(adapt_out).view(2,256*512).size())

        ##### out1 : real image patch passed thru Fusion net , size -> (batch,2)
        ##### out2 : adapt image patch passed thru Fusion net , size -> (batch,2)

        ###### creating good and bad label for loss calculation #######
        # pdist = nn.PairwiseDistance(p=2)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        bs = torch.from_numpy(real_out).size(0)
        # distance_real = pdist(torch.from_numpy(real_out).view(2,256*512).type(torch.FloatTensor),target_real.view(2,256*512).type(torch.FloatTensor))
        # distance_adapt = pdist(torch.from_numpy(adapt_out).view(2,256*512).type(torch.FloatTensor),target_real.view(2,256*512).type(torch.FloatTensor))
        # print(type(real_out),type(adapt_out),type(target_real))
        # print(distance_real)
        distance_real = cos(torch.from_numpy(real_out_max).type(torch.FloatTensor).view(bs,256*512),target_real.type(torch.FloatTensor).view(bs,256*512))
        distance_adapt = cos(torch.from_numpy(adapt_out).type(torch.FloatTensor).view(bs,256*512),target_real.type(torch.FloatTensor).view(bs,256*512))
        max_sim = torch.max(distance_real, distance_adapt)
        # min_dist = cos(torch.from_numpy(real_out).type(torch.FloatTensor).view(bs,256*512),torch.from_numpy(adapt_out).type(torch.FloatTensor).view(bs,256*512) )
        ############# max sim gives the similarity value per batch which is more closer to GT #######
        # print("max distance", max_sim)
        
        
        # print("the cosine distance is", min_dist)

        # print(distance_real,distance_adapt)

        ############ label_real : gives if max value came from real tensor or adapt tensor #########
        label_real = torch.eq(distance_real, max_sim)
        label_adapt = torch.eq(distance_adapt, max_sim)
        # print(label_real,label_adapt)
        
        ############# assign the good label closest to GT as label ############
        # tor_list=np.array([],dtype=np.float32)
        tor_list = np.zeros((bs,1),dtype=np.float32)
        # print(tor_list)
        # result_array = np.array([])
        # lis = [tor_list.append(i) for i in label_real]
        # print(lis)
        temp_list= []
        for i in label_real:
            # print(i)
            temp_list.append([int(i.data[0])])
        np.append(tor_list,np.array(temp_list), axis=0)
        # print("torch list",tor_list)
        # one_hot = torch.stack((label_real, label_adapt), 1)

        # print(one_hot)
        # print("before transpose", one_hot.size())
        # print("after transpose", torch.transpose(one_hot,1,0).size())
        # # print(label_real)
        # print(type(out1),type(out2),type(torch.transpose(one_hot,1,0).byte()))
        # from_numpy(np.array([int(self.training_df.iat[index,2])],dtype=np.float32)
        lab = torch.from_numpy(tor_list)
        lab.requires_grad = True
        L_fuse = fusion_loss(out1,out2,lab)
        # L_fuse.requires_grad = True
        # print("pred size"+ str(torch.from_numpy(real_out).size()) + "tar size"+ str(target_real.size()))
        real_out = torch.from_numpy(real_out).type(torch.FloatTensor)
        real_out.requires_grad = True
        # target_real.requires_grad = True
        adapt_out = adapt_out_max.type(torch.FloatTensor)
        adapt_out.requires_grad = True
        L_real = real_loss(real_out,target_real) + mse_loss(torch.from_numpy(real_out.detach().max(dim=1)[1].cpu().numpy()).type(torch.FloatTensor), target_real.type(torch.FloatTensor))
        # L_real.requires_grad=True

        print(adapt_out_max.size())
        print(target_real.size())
        L_adapt = adapt_loss(adapt_out, target_real) + jacc_loss(torch.from_numpy(adapt_out.detach().max(dim=1)[1].cpu().numpy()).type(torch.FloatTensor),target_real.type(torch.FloatTensor))
        # L_adapt.requires_grad = True
        L_real.backward(retain_graph= True)
        L_adapt.backward(retain_graph= True)
        L_mul = L_real + L_adapt
        L_mul.backward()
        # print(L_adapt)
        optimizer_fuse.step()
        L_tot =  L_adapt + L_fuse + L_real
        # print("Total Loss", L_tot)
            
        # total_loss = [L_adapt,L_real,L_fuse]
        # L_end2end = torch.mean(L_adapt+L_real,L_fuse)
        # L_end2end.backward()
        real_optim.step()

        real_schedule.step()
        np_loss = L_tot.detach().cpu().numpy()
        interval_loss += np_loss
        if (cur_itrs) % bs == 0:
                interval_loss = interval_loss/bs
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, total_iter, interval_loss))
                interval_loss = 0.0

############ All Loss will the entwork end to end and also flow into Real CHannel ###############
############ During Test time we will only use Real Channel Output So Saving Real Weights #########
        
        if (cur_epochs) % 10 == 0:
            save_model_ckpt(cfg.model_ckpt+'/best_%s_%s_os%d.pth' %
                              ('deeplabv3plus_mobilenet', 'cityscapes',output_stride), real, cur_itrs)
        # print(L_real)
        
        # print("real nisenet otput ", out1.size())




        # print(image.size())
        # print(image_real.size())
        # image = image.to("cpu")



