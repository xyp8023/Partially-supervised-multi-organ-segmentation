import matplotlib.pyplot as plt
import torch

class To_Class(object):
    def __init__(self):
        self.background = [0,0,0] # [1,0,0,0]
        self.crop = [102,255,102]# [0,1,0,0]
        self.soil = [51,221,255]# [0,0,1,0]
        self.thistle = [250,50,183]# [0,0,0,1]

    def call(self, label):
        # input: label, shape (N,3,h,w)
        # return label_onehot, shape (N,1,h,w)
        N,c,h,w = label.shape
        mask_background = label[:,0,:,:]==self.background[0]
        mask_crop = label[:,0,:,:]==self.crop[0]
        mask_soil = label[:,0,:,:]==self.soil[0]
        mask_thistle = label[:,0,:,:]==self.thistle[0]

        label_onehot = torch.zeros((N, h, w))
        # print(mask_background.shape)
        label_onehot[mask_background]=0

        label_onehot[mask_crop]=1

        label_onehot[mask_soil]=2

        label_onehot[mask_thistle]=3

        return label_onehot


class To_Onehot(object):
    def __init__(self):
        self.background = [0,0,0] # [1,0,0,0]
        self.crop = [102,255,102]# [0,1,0,0]
        self.soil = [51,221,255]# [0,0,1,0]
        self.thistle = [250,50,183]# [0,0,0,1]

    def call(self, label):
        # input: label, shape (N,3,h,w)
        # return label_onehot, shape (N,4,h,w)
        N,c,h,w = label.shape
        mask_background = label[:,0,:,:]==self.background[0]
        mask_crop = label[:,0,:,:]==self.crop[0]
        mask_soil = label[:,0,:,:]==self.soil[0]
        mask_thistle = label[:,0,:,:]==self.thistle[0]

        label_onehot = torch.zeros((N, 4, h, w))
        # print(mask_background.shape)
        label_onehot[:,0,:,:][mask_background]=1
        label_onehot[:,1,:,:][mask_background]=0
        label_onehot[:,2,:,:][mask_background]=0
        label_onehot[:,3,:,:][mask_background]=0

        label_onehot[:,0,:,:][mask_crop]=0
        label_onehot[:,1,:,:][mask_crop]=1
        label_onehot[:,2,:,:][mask_crop]=0
        label_onehot[:,3,:,:][mask_crop]=0

        label_onehot[:,0,:,:][mask_soil]=0
        label_onehot[:,1,:,:][mask_soil]=0
        label_onehot[:,2,:,:][mask_soil]=1
        label_onehot[:,3,:,:][mask_soil]=0

        label_onehot[:,0,:,:][mask_thistle]=0
        label_onehot[:,1,:,:][mask_thistle]=0
        label_onehot[:,2,:,:][mask_thistle]=0
        label_onehot[:,3,:,:][mask_thistle]=1

        return label_onehot


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()