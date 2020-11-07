import argparse
import os

import numpy as np
import torch

from skimage.io import imsave
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensor
from torchvision import transforms
from tqdm import tqdm
import cv2

from common. dataset import MedicalImageDataset as Dataset
from common.loss import bce_dice_loss,  dice_coef_metric
from model.Att_Unet import Att_Unet
from common.utils import log_images, gray2rgb,  outline


def main(config):

    makedirs(config)
    device = torch.device("cpu" if not torch.cuda.is_available() else config.device)
   # logger = Logger(config.logs)
    loader,dataset = data_loader(config)

    with torch.set_grad_enabled(False):
        unet = Att_Unet()
        state_dict = torch.load(config.weights, map_location=device)
        unet.load_state_dict(state_dict)
        unet.eval()
        unet.to(device)



        input_list = []
        pred_list = []
        #true_list = []
        #loss_test = []
        step=0

        for i, data in tqdm(enumerate(loader)):
            step += 1
            x= data
            x= x.to(device)

            y_pred = unet(x)

            #loss = bce_dice_loss(y_pred, y_true)
            #loss_test.append(loss.item())

            y_pred_np = y_pred.detach().cpu().numpy()
            pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

            #y_true_np = y_true.detach().cpu().numpy()
            #true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

            x_np = x.detach().cpu().numpy()
            input_list.extend([x_np[s] for s in range(x_np.shape[0])])

        if config.mask_outline==True:

            for i in range(len(input_list)):
                image = gray2rgb(np.squeeze(input_list[i][0,:,:]))
                image = outline(image, pred_list[i][0,:,:], color=[255, 0, 0])
                image=image.astype("uint8")
                filename="{}.png".format(i)
                filepath = os.path.join(config.predictions, filename)
                imsave(filepath, image)

        else:
            for i in range(len(input_list)):
                 img_name=dataset.imgs[i].split("/" )[-1:][0]
                 org_shape=cv2.imread(dataset.imgs[i]).shape
                 mask=pred_list[i][0,:,:]
                 mask[np.nonzero(mask<0.3)]=0.0
                 mask[np.nonzero(mask>0.3)]=255.0
                 mask=mask.astype("uint8")
                 mask_copy=mask.copy()
                 mk=cv2.resize(mask,(org_shape[0],org_shape[1]),interpolation = cv2.INTER_AREA)


                 filename="{}".format(img_name)
                 filepath = os.path.join(config.predictions, filename)
                 imsave(filepath, mk)








def makedirs(config):
    os.makedirs(config.predictions, exist_ok=True)

data_transforms = A.Compose ([
    A.Resize(width = 256, height = 256, p=1.0),
    A.Normalize( p=1.0),
    ToTensor()
])


def data_loader(config):
    dataset = Dataset('test', config.root,
                    transform=data_transforms
                    )

    loader =DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=4
    )

    return loader,dataset




"""
def compute_iou(model, loader, threshold=0.3):

    Computes accuracy on the dataset wrapped in a loader
    Returns: accuracy as a float value between 0 and 1

    device = torch.device("cpu" if not torch.cuda.is_available() else config.device)
    #model.eval()
    valloss = 0

    with torch.no_grad():

        for i_step, (data, target) in enumerate(loader):

            data = data.to(device)
            target = target.to(device)


            #prediction = model(x_gpu)

            outputs = model(data)
           # print("val_output:", outputs.shape)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            picloss = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valloss += picloss

        #print("Threshold:  " + str(threshold) + "  Validation DICE score:", valloss / i_step)

    return valloss / i_step

"""

"""
def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)
"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for segmentation of polyps in GI"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights/unet.pt", required=True, help="path to weights file"
    )
    parser.add_argument(
        "--root", type=str, default="./medico2020", help="root folder with images"
    )

    parser.add_argument(
        "--mask_outline", type=bool, default=False, help="If True ,draws border line of mask else returns binary mask"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="./predictions",
        help="folder for saving images with prediction outlines",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=20,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )

    parser.add_argument(
        "--logs", type=str, default="./test_logs", help="folder to save logs"
    )

    config = parser.parse_args()
    main(config)
