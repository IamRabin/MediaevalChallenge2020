import argparse
import os

import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import MedicalImageDataset as Dataset
from logger import Logger
from loss import DiceLoss, dice_coef_metric
from unet_pre import unet_pre
from utils import log_images


def main(config):

    device = torch.device("cpu" if not torch.cuda.is_available() else config.device)
    logger = Logger(config.logs)
    loader = data_loader(config)

    with torch.set_grad_enabled(False):
        unet = unet_pre()
        state_dict = torch.load(config.weights, map_location=device)
        unet.load_state_dict(state_dict)
        unet.eval()
        unet.to(device)
        dsc_loss = DiceLoss()


        input_list = []
        pred_list = []
        #true_list = []
        #loss_test = []
        step=0

        for i, data in tqdm(enumerate(loader)):
            step += 1
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = unet(x)

            #loss = dsc_loss(y_pred, y_true)
            #loss_test.append(loss.item())

            y_pred_np = y_pred.detach().cpu().numpy()
            pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

            #y_true_np = y_true.detach().cpu().numpy()
            #true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

            x_np = x.detach().cpu().numpy()
            input_list.extend([x_np[s] for s in range(x_np.shape[0])])


            if i * config.batch_size < config.vis_images:
                tag = "image/{}".format(i)
                num_images = config.vis_images - i * config.batch_size
                logger.image_list_summary(
                    tag,
                    log_images(x,y_true=False,  y_pred)[:num_images],
                    step,
                )


    #    log_loss_summary(logger,  step, prefix="test_")
    #    mean_dsc = compute_iou(unet,loader)
    # logger.scalar_summary("test_dsc", mean_dsc, step)
        #loss_test = []

        # print("Test mean DSC: {:4f}".format(mean_dsc))




data_transforms = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#mask_transform = transforms.Compose([
                              # transforms.Resize((256, 256)),
                               #transforms.ToTensor() ])


def data_loader(config):
    dataset = Dataset('test', config.root,
                    transform=data_transforms,
                    mask_transform=None,
                    augment=True,
                    equalize=True)

    loader =DataLoader(
        dataset,
        batch_size=config.batch_size,
        drop_last=False,
        num_workers=4
    )

    return loader


"""
def compute_iou(model, loader, threshold=0.3):
    """
    Computes accuracy on the dataset wrapped in a loader

    Returns: accuracy as a float value between 0 and 1
    """
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
        "--weights", type=str, required=True, help="path to weights file"
    )
    parser.add_argument(
        "--root", type=str, default="./medico2020", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
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
