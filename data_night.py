import cv2
import numpy as np 
import os

data_path = 'data_' ############### path containing cityscapes data #########
split = "val" ######## ['train','test','val'] ########
images_dir = os.path.join(data_path, 'leftImg8bit', split)

gamma = 0.35
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")


for city in os.listdir(images_dir):
    img_dir = os.path.join(images_dir, city)
    for imgs in os.listdir(img_dir):
        img_name = os.path.join(img_dir, imgs)
        im = cv2.imread(img_name)
        im = im[:,:,::-1].copy()
        im_save = cv2.LUT(im, table)
        cv2.imwrite(img_name,im_save)
        
