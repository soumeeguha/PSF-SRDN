import torch
import torch.nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn.functional import normalize
from PIL import Image
import glob
import random
import matplotlib.pyplot as plt 
import cv2

def get_train_test_image_lists(root, folders = False):

    if folders is False:
        train_img = glob.glob(root + 'train/*.png')
        test_img = glob.glob(root + 'test/*.png')
       
    else:
        folders = glob.glob(root + '*')
        train_img, test_img = [], []
        for folder in folders:
            # print(folder)
            train_img.extend(glob.glob(folder + '/train/*.png'))
            test_img.extend(glob.glob(folder + '/test/*.png'))
        

    
    return train_img, test_img

def get_phantom_train_test_image_lists(root, train_ratio = 0.7, num_imgs = 100):
    folders = glob.glob(root + '*')
    for i in range(len(folders)):
        folders[i] = folders[i].split('/')[-1]
    train_indices = sorted(random.sample(range(1, num_imgs+1), int((1 - train_ratio)*num_imgs)))
    test_indices = list(set(list(range(1, num_imgs+1))) - set(train_indices))

    train_img, test_img = [], []

    for i in range(num_imgs):
        for folder in folders:
            if i+1 in train_indices:
                train_img.append(root + folder + f'/img_{i+1}.png')
            else:
                test_img.append(root + folder + f'/img_{i+1}.png')


    train_label, test_label = [], []
    
    return train_img, train_label, test_img, test_label


def get_test_image_list(root, folders):

    images = []
    for folder in folders:
        images.extend(glob.glob(root + folder + '/*.tiff'))
        # print(len(images))
    return images, []




class get_Dataset(Dataset):
    def __init__(self, image_size, image_list, transform = None, if_print = None):
        super().__init__()
        self.image_list = image_list
        self.transform = transform
        self.image_size = image_size
        self.toTensor = transforms.ToTensor()
        self.print = if_print


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        im = cv2.imread(str(self.image_list[index]), cv2.IMREAD_GRAYSCALE)/255
        image = self.toTensor(im).type(torch.float)
        if self.transform is not None:
            image = self.transform(image)
        if self.print is None:
            return image
        else:
            folder = self.image_list[index].split('/')[-2]
            return self.image_list[index].split('/')[-1].split('.')[0], image

def setup_dataloader(batch_size, num_worker_train, train_ds, test_ds=None):
    """Set's up dataloader"""

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=num_worker_train,
        pin_memory=True)

    if test_ds is not None:

        test_dl = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=num_worker_train,
            pin_memory=False)
    else:

        test_dl = None

    return train_dl, test_dl
