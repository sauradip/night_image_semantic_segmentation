[![DOI](https://zenodo.org/badge/208427167.svg)](https://zenodo.org/badge/latestdoi/208427167)


# What's There in the Dark, ICIP19 ( ICIP "Spotlight Paper" )

This is the official github repository for the paper "What's There in The Dark" accepted in IEEE International Conference in Image Processing 2019 (ICIP19) , Taipei, Taiwan. [ [Paper](https://ieeexplore.ieee.org/abstract/document/8803299/authors#authors) ][ [Papers with code]( https://paperswithcode.com/paper/whats-there-in-the-dark) ]

## Abstract 

Scene Parsing is an important cog for modern autonomous driving systems. Most of the works in semantic segmentation pertains to day-time scenes with favourable weather and illumination conditions. In this paper, we propose a novel deep architecture, NiSeNet, that performs semantic segmentation of night scenes using a domain mapping approach of synthetic to real data. It is a dual-channel network, where we designed a Real channel using DeepLabV3+ coupled with an MSE loss to preserve the spatial information. In addition, we used an Adaptive channel reducing the domain gap between synthetic and real night images, which also complements the failures of Real channel output. Apart from the dual channel, we introduced a novel fusion scheme to fuse the outputs of two channels. In addition to that, we compiled a new dataset Urban Night Driving Dataset (UNDD); it consists
of 7125 unlabelled day and night images; additionally, it has 75 night images with pixel-level annotations having classes equivalent to Cityscapes dataset. We evaluated our approach on the Berkley Deep Drive dataset, the challenging Mapillary dataset and UNDD dataset to exhibit that the proposed method outperforms the state-of-the-art techniques in terms of accuracy and visual quality

## Demo 

![Demo](https://github.com/sauradip/night_image_semantic_segmentation/blob/master/images/others/demo_video.gif)

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

