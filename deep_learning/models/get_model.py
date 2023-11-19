import torch


def get_model(MODEL, IMAGE_SIZE):
    if MODEL == "UNet":
        from unet import UNet
        model = UNet(outSize=(IMAGE_SIZE, IMAGE_SIZE), )

    elif MODEL == "SegResNet":
        from monai.networks.nets import SegResNet

        model = SegResNet(in_channels=1, out_channels=2, spatial_dims=2)

    elif MODEL == "MobileNetV3":
        from torchvision import models
        model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=False, num_classes=2)
        model.backbone["0"][0] = torch.nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        model
    elif MODEL == "ResUNEt":
        from models.res_unet import unet

        model = unet(in_channels=1, out_channels=2)

    elif MODEL == "MobileViTV2":
        from transformers import MobileViTV2ForSemanticSegmentation, MobileViTV2Config

        configuration = MobileViTV2Config(num_channels=1, image_size=IMAGE_SIZE, num_classes=2, return_dict=False, output_stride=8)
        model = MobileViTV2ForSemanticSegmentation(configuration)

    elif MODEL == "Segformer":
        from transformers import  SegformerForSemanticSegmentation, SegformerConfig


        configuration = SegformerConfig(num_channels=1, image_size=IMAGE_SIZE, num_classes=2, return_dict=False)
        model = SegformerForSemanticSegmentation(configuration)

    else:
        raise Exception("Model not implemented")
    
    return model