from __future__ import print_function, division

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

# Ignore warnings
import warnings
import pdb

warnings.filterwarnings("ignore")


def make_dataset(root,mode):

  """   Takes in the root directory and mode(train or val or test) as inputs
  then joins the path with the folder of the specified mode.Applies normalize
  function to each img of the folder and returns a list of tuples containing
  image and its corresponding mask.

  Returns
  -------
  tuple : Normalized image and its annotation.

  """
  assert mode in ['train', 'val', 'test']
  items = []

  if mode == 'train':
      train_img_path = os.path.join(root, 'Train/train_image')
      train_mask_path = os.path.join(root, 'Train/train_mask')

      images = os.listdir(train_img_path)
      labels = os.listdir(train_mask_path)

      images.sort()
      labels.sort()

      for it_im, it_gt in zip(images, labels):
          item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
          items.append(item)
  elif mode == 'val':
      val_img_path = os.path.join(root, 'Val/val_img')
      val_mask_path = os.path.join(root, 'Val/val_mask')

      images = os.listdir(val_img_path)
      labels = os.listdir(val_mask_path)

      images.sort()
      labels.sort()


      for it_im, it_gt in zip(images, labels):
          item = (os.path.join(val_img_path, it_im), os.path.join(val_mask_path, it_gt))
          items.append(item)
  else:
      test_img_path = os.path.join(root, 'Test/test_img')
     # test_mask_path = os.path.join(root, 'Test/test_mask')

      images = os.listdir(test_img_path)
      #labels = os.listdir(test_mask_path)

      images.sort()
      #labels.sort()

      for it_im in images:
          item = os.path.join(test_img_path, it_im)
          items.append(item)

  return items



class MedicalImageDataset(Dataset):
    """ GI dataset."""

    def __init__(self, mode, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.mode=mode
        self.transform = transform
        self.imgs = make_dataset(root_dir, mode)


    def __len__(self):
        return len(self.imgs)



    def __getitem__(self,index):
      if self.mode== 'test':
         img_path=self.imgs[index]
         img =np.array( Image.open(img_path))
         img_shape=np.array(img).shape

         if self.transform:
            augmented = self.transform(image=img)
            image=augmented["image"]


         return image

      else:
        img_path, mask_path = self.imgs[index]
        # print("{} and {}".format(img_path,mask_path))
        img = np.array(Image.open(img_path))  # .convert('RGB')
        # mask = Image.open(mask_path)  # .convert('RGB')
        # img = Image.open(img_path).convert('L')
        mask = np.array(Image.open(mask_path).convert('L'))

        # print('{} and {}'.format(img_path,mask_path))

        if self.transform:
            augmented = self.transform(image=img,mask=mask)
            image=augmented["image"]
            mask=augmented["mask"]

        return [image, mask]
