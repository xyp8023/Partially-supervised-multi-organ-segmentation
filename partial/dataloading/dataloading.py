import numpy as np
import imageio
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from PIL import Image
import random

class ColorMap(object):
    """
    # label:color_rgb:parts:actions
background:0,0,0:: 0
crop:102,255,102:: 1
soil:51,221,255:: 2
thistle:250,50,183:: 3
    """
    def __init__(self):
        self.classes = ["background", "crop", "soil", "thistle"]
        self.colormap = [[0,0,0], [102,255,102], [51,221,255], [250,50,183]]
        self.cm2lbl = np.zeros(256**3)
        for i,cm in enumerate(self.colormap):
            self.cm2lbl[(cm[0]*256+cm[1])*256+cm[2]]=i
    def img2label(self, im, data_type="array"):
        if data_type=="array":
            data = im.astype("int32")
            idx = (data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]
            return np.array(self.cm2lbl[idx])
        elif data_type=="tensor":
            # print("im in ColorMap.img2label: ", im.shape)
            # np.save("./im.npy", im.cpu().numpy())
            # print(im.dtype, im.max())
            im = (im.numpy().transpose((1, 2, 0))*255).astype(np.uint8) # C x H x W -> H x W x C
            data = im.astype("int32")
            idx = (data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]
            # data = np.array(self.cm2lbl[idx]) 
            # print("np.unique: ", np.unique(np.array(self.cm2lbl[idx])))    
            # assert 1==0
            return torch.from_numpy(np.array(self.cm2lbl[idx])).long()
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
    def __init__(self, th, tw):
        self.size = (th, tw)
    
    def __call__(self, *images):
        # perform some check to make sure images are all the same size
        # if self.padding > 0:
            # images = [ImageOps.expand(img, border=self.padding, fill=0) for im in images]

        w, h = images[0].size
        th, tw = self.size
        if w == tw and h == th:
            return images

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in images]
    
class WeedCropDataset(Dataset):

    """Weed Crop dataset.

    # label:color_rgb:parts:actions
    background:0,0,0::
    crop:102,255,102::
    soil:51,221,255::
    thistle:250,50,183::

    """

    def __init__(self, root_dir, transform=ToTensor(),th=512,tw=512, img_file="img_name1.txt", label_file="label_name1.txt"):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        img_csv_file = root_dir + os.sep + img_file
        label_csv_file = root_dir + os.sep + label_file

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
        
        self.cm = ColorMap()
        
        self.rc = RandomCrop(th, tw)
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

        
        sample = {'img': img, 'label': label, "img_name": img_name, "label_name": label_name}

        if self.transform:
            sample["img"], sample["label"] = self.rc(sample["img"], sample["label"])
            sample["img"] = self.transform(sample["img"])
            sample["label"] = self.transform(sample["label"]) # torch.Size([3, 256, 256])
            # print(sample["label"].shape)
            sample["label"] = self.cm.img2label(sample["label"], data_type="tensor")
            # print(sample["label"].shape)
        # rc = RandomCrop()    
        # sample = RandomCrop.__call__(sample)
        return sample
