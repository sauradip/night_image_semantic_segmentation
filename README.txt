# What's There in the Dark

Pytorch implementation of our method for Night Image Semantic Segmentation

Contact: Sauradip Nag ( Email : sauradipnag95 at gmail dot com)

## Environment 

* The requirements file icip_requirements.txt is provided. It will run on python 3.6.10 version. 

* Make a pip3 virtual environment with the requirements file to run the code to avoid version conflicts. 

## To Train the code

* Step1 : Preprocess the data -> Sample data has been provided in "data_" folder. However, to create the night images, one has to go into the folder "preprocessing" and then run "test_cyclegan.sh" to generate night images for corresponding day images. The pretrained model for night images that i have trained is provided in this folder. 

* Step2 :  Arrange the night images from preprocessing step in the format of target dataset( here "cityscapes") as shown in example "data_" folder, i.e Night Images in "leftImg8bit" folder and GT In "gtFine" folder. 

* Step3 : If you want to run for other datasets, follow step1 and step2 for target dataset and train the real model ( in "real" folder) and synthetic model ( in "adapt" folder) separately for night samples. The training codes for each channels are given in their respective folder. For Cityscapes dataset, I have given the trained models ( *** however i could not find the last model for adapt channel which i used for training, so i provided a old pretrained model on cityscapes *** )

* Step4 : Set Paths and other hyperparameters in file  config/cityscapes_config.py for training. 

* Step5 : To start training run "main_model.py", the resulting checkpoint will be saved in "real_checkpoint" folder. 

## To Test the code 

* Step1 : The checkpoint model stored in "real_checkpoint" folder will be used in testing. I have already provided a existing pretrained model for NiSeNet.

* Step2 : The mode needs to be decided, whether "indoor" or "outdoor", then comment/uncomment the appropriate python code for the mode in file "test_v2.sh" and run it. Results will be stored in folder "results" in home directory. 


## Sample Results

Contains data format / output of few things :

* preprocessing : contains output format produced by cycle_gans trained for night images 

* cityscapes_night_format : contains the night images obtained from preprocessing step into cityscapes dataset format

* cityscapes_output : contains few samples of the output of our NiSeNet model on the testing split of cityscapes. 

* indoor_output : contains few output samples on the indoor dataset


