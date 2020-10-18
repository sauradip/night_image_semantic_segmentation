config = {

    ######## Outdoor Model Configuration ########
     
    "outdoor_path" : ,
    "outdoor_model" : deeplabv3plus_mobilenet ,
    "lr" : 0.01,
    "crop_size" : 224,
    "batch_size" : 1,
    "output_stride" : 16,
    "outdoor_checkpoint" : real/DeepLabV3Plus-Pytorch/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth,
    "outdoor_save_path" : , 

    ######## Indoor Model Configuration ########

    "indoor_path" : ./indoor/img_data,
    "indoor_model_cfg" : indoor/config/ade20k-mobilenetv2dilated-c1_deepsup.yaml ,
    "indoor_model" : ade20k-mobilenetv2dilated-c1_deepsup , 
    "outdoor_checkpoint" : indoor/epoch_20.pth,
    "outdoor_save_path" : ,

}

test_mode = "outdoor"    ## indoor or outdoor ?

if test_mode == "outdoor":
    python3 real/DeepLabV3Plus-Pytorch/main.py --model config["outdoor_model"] \
    --crop_val --lr config["lr"] --crop_size config["crop_size"] --batch_size config["batch_size"] \
    --output_stride config["output_stride"] --ckpt config["outdoor_checkpoint"] \
    --test_only --save_val_results

elif test_mode == "indoor":
    python3 -u indoor/test.py \
    --imgs config["indoor_path"] \
    --cfg config["indoor_model_cfg"] \
    DIR config["indoor_model"] \
    TEST.result config["outdoor_save_path"] \
    TEST.checkpoint config["outdoor_checkpoint"]