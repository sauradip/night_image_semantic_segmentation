# What's There in the Dark

This is the official github repository for the paper "What's There in The Dark" accepted in IEEE International Conference in Image Processing 2019 (ICIP19) , Taipei, Taiwan. [ [Paper](https://ieeexplore.ieee.org/abstract/document/8803299/authors#authors) ]

### Model Architecture : 

![Model Architecture](https://github.com/sauradip/night_image_semantic_segmentation/blob/master/images/others/archi.jpg)

Since we are submitting in journal, we currently cannot make the code public, however, we are making the data preparation code public. Infact we are the first one to make the model for day to night image conversion public. However, we have made the multi-scale architecture code public. Interested users can download adaptsegnet and deeplabv3+ and plug this code in as the last module and train. 

### Day to Night Conversion using CycleGANS ( [CycleGANS](https://github.com/junyanz/CycleGAN) )

##### Steps : 
- Go to the CycleGANS github link and clone the code.
- Download the pretrained model from the following link [ [Pretrained_Model](https://drive.google.com/open?id=1B7KvOMZI1nMkcuUrXrZnNZ0ptG2vs7Tp) ].
- Place the folder contents (latest_net_G_A.pth, latest_net_G_B.pth, latest_net_D_A.pth, latest_net_D_B.pth) into the checkpoint/any_name folder.
- Run the testing code as mentioned in CycleGANS website.

## Result on Berkley Deep Drive Dataset

<p align="center">
  <img src="https://github.com/sauradip/night_image_semantic_segmentation/blob/master/images/bdd/merge.png">
</p>

## Result on Cityscapes Dataset

<p align="center">
  <img src="https://github.com/sauradip/night_image_semantic_segmentation/blob/master/images/cityscapes/merge.png">
</p>

## Result on Mapillary Dataset

<p align="center">
  <img src="https://github.com/sauradip/night_image_semantic_segmentation/blob/master/images/mapillary/merge.png">
</p>

# Results 

The results were generated using NiSeNet on the following dataset BDD, Mapillary, UNDD(proposed) : 

<p align="center">
  <img src="https://github.com/sauradip/night_image_semantic_segmentation/blob/master/images/others/comp_result2.png">
</p>


# References 

If you find this code useful in your research, please consider citing:

> @inproceedings{nag2019s,
  title={What’s There in the Dark},
  author={Nag, Sauradip and Adak, Saptakatha and Das, Sukhendu},
  booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
  pages={2996--3000},
  year={2019},
  organization={IEEE}
}

