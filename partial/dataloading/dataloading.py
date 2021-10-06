import numpy as np
import imageio
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from PIL import Image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = img.transpose((2, 0, 1))

        return torch.from_numpy(img)
#
# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         img, label = sample['img'], sample['label']

#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C x H x W
#         img = img.transpose((2, 0, 1))
#         label = label.transpose((2, 0, 1))

#         return {'img': torch.from_numpy(img),
#                 'label': torch.from_numpy(label)}

class RandomCrop(object):
    def __call__(self,sample):
        img, label = sample['img'], sample['label']

        rc = transforms.RandomCrop(256)
        img = rc(img)
        label = rc(label)
        return {'img': img,
                'label': label}

class WeedCropDataset(Dataset):

    """Weed Crop dataset.

    # label:color_rgb:parts:actions
    background:0,0,0::
    crop:102,255,102::
    soil:51,221,255::
    thistle:250,50,183::

    """

    def __init__(self, root_dir, transform=ToTensor()):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        img_csv_file = root_dir + os.sep + "img_name.txt"
        label_csv_file = root_dir + os.sep + "label_name.txt"

        with open(img_csv_file) as the_file:
            self.img_names = the_file.readlines()
        with open(label_csv_file) as the_file:
            self.label_names = the_file.readlines()

        
        self.root_dir = root_dir
        self.transform = transform
        self.background = [0,0,0] # [1,0,0,0]
        self.crop = [102,255,102]# [0,1,0,0]
        self.soil = [51,221,255]# [0,0,1,0]
        self.thistle = [250,50,183]# [0,0,0,1]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
            # idx = idx.tolist()

        img_name = self.img_names[idx].split()[0]
        # img = io.imread(img_name)
        img = Image.open(img_name).convert("RGB")
        label_name = self.label_names[idx].split()[0]
        # label = io.imread(label_name)
        label = Image.open(label_name).convert("RGB")
        # mask_background = label[:,:,0]==self.background[0]
        # mask_crop = label[:,:,0]==self.crop[0]
        # mask_soil = label[:,:,0]==self.soil[0]
        # mask_thistle = label[:,:,0]==self.thistle[0]

        # label_onehot = np.zeros((label.height, label.width, 4))
        # label_onehot[mask_background,:]=[1,0,0,0]
        # label_onehot[mask_crop,:]=[0,1,0,0]
        # label_onehot[mask_soil,:]=[0,0,1,0]
        # label_onehot[mask_thistle,:]=[0,0,0,1]

        
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'img': img, 'label': label, "img_name": img_name, "label_name": label_name}

        if self.transform:
            sample["img"] = self.transform(sample["img"])
            sample["label"] = self.transform(sample["label"])

        return sample
