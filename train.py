import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import MedicalImageDataset as Dataset
from logger import Logger
from loss import bce_dice_loss, dice_coef_metric
from Att_Unet import Att_Unet
from utils import log_images


def main(config):
    makedirs(config)
    snapshotargs(config)
    device = torch.device("cpu" if not torch.cuda.is_available() else config.device)

    loader_train, loader_valid = data_loaders(config)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet =Att_Unet()
    unet.to(device)

    
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=config.lr,weight_decay=1e-5)
    lr_scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=False)


    logger = Logger(config.logs)
    loss_train = []
    loss_valid = []

    step = 0

    for epoch in tqdm(range(config.epochs), total=config.epochs):
        for phase in ["train", "valid"]:
                if phase == "train":
                    unet.train()
                else:
                    unet.eval()

                validation_pred = []
                validation_true = []

                for i, data in enumerate(loaders[phase]):
                    if phase == "train":
                        step += 1

                    x, y_true = data
                    x, y_true = x.to(device), y_true.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                            y_pred = unet(x)

                            loss = bce_dice_loss(y_pred, y_true)

                            if phase == "valid":
                                loss_valid.append(loss.item())
                                y_pred_np = y_pred.detach().cpu().numpy()
                                validation_pred.extend(
                                    [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                                )
                                y_true_np = y_true.detach().cpu().numpy()
                                validation_true.extend(
                                    [y_true_np[s] for s in range(y_true_np.shape[0])]
                                )
                                if (epoch % config.vis_freq == 0) or (epoch == config.epochs - 1):
                                    if i * config.batch_size < config.vis_images:
                                        tag = "image/{}".format(i)
                                        num_images = config.vis_images - i * config.batch_size
                                        logger.image_list_summary(
                                            tag,
                                            log_images(x, y_true, y_pred)[:num_images],
                                            step,
                                        )

                            if phase == "train":
                                loss_train.append(loss.item())
                                loss.backward()
                                optimizer.step()

                    if i % 50 == 0:
                        for param_group in optimizer.param_groups:
                            print("Current learning rate is: {}".format(param_group['lr']))
                            print("Epoch[{}/{}]({}/{}): Loss: {:.4f}".format(epoch+1, config.epochs, i, len(loaders["train"]), loss.item()))

                                

                    if phase == "train" and (step + 1) % 10 == 0:
                        log_loss_summary(logger, loss_train, step)
                        loss_train = []

                if phase == "valid":
                    log_loss_summary(logger, loss_valid, step, prefix="val_")
                    mean_dsc = compute_iou(unet,loaders[phase])
                    logger.scalar_summary("val_dsc", mean_dsc, step)
                    lr_scheduler.step(mean_dsc)
                    print("\nMean DICE on validation:", mean_dsc)
                    
                    if mean_dsc > best_validation_dsc:
                        best_validation_dsc = mean_dsc
                        torch.save(unet.state_dict(), os.path.join(config.weights, "unet.pt"))
                    loss_valid = []



    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))


def data_loaders(config):
    dataset_train, dataset_valid = datasets(config)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=config.batch_size,
        drop_last=False,
        num_workers=config.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid



data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

mask_transform = transforms.Compose([
                               transforms.Resize((256, 256)),
                               transforms.ToTensor() ])


def datasets(config):
    train = Dataset('train', config.root,
                    transform=data_transforms,
                    mask_transform=mask_transform,
                    augment=True,
                    equalize=True)




    valid = Dataset('val', config.root,
                    transform=data_transforms,
                    mask_transform=mask_transform,
                    equalize=True)
    return train, valid


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


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(config):
    os.makedirs(config.weights, exist_ok=True)
    os.makedirs(config.logs, exist_ok=True)


def snapshotargs(config):
    config_file = os.path.join(config.logs, "config.json")
    with open(config_file, "w") as fp:
        json.dump(vars(config), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetuning pretrained Unet"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=100,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
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
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=6,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    config = parser.parse_args()
    main(config)
