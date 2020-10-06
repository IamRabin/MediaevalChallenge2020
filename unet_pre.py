
import torch
import torch.nn as nn




def unet_pre():
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',in_channels=3,
                        out_channels=1, init_features=32, pretrained=True,verbose=False)
    return model
