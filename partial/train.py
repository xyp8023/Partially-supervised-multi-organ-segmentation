import torch
import numpy as np
from dataloading.dataloading import WeedCropDataset, ToTensor, RandomCrop
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from unet import UNet
from utils.dice_score import dice_loss
from utils.utils import To_Onehot, To_Class

import torch.nn as nn
import torch.nn.functional as F
from torch import optim


## hyper-parameters
learning_rate=0.002
amp=False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transforms = transforms.Compose([ transforms.RandomCrop(256), 
                                       transforms.ToTensor()]) 


dataset = WeedCropDataset(root_dir="datasets/debug",transform=train_transforms)
# dataset = WeedCropDataset(root_dir="datasets/val")

dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=2)

net = UNet(n_channels=3, n_classes=4, bilinear=True)
net.to(device=device)
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
# grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
criterion = nn.CrossEntropyLoss()

# to_onehot = To_Onehot()
to_onehot = To_Class()


num_epochs = 40
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        source = sample_batched['img'].to(device=device, dtype=torch.float32) # (N,3,256,256)
        target = (sample_batched['label']*255).to(device=device, dtype=torch.uint8)# (N,3,256,256)
        target_ = to_onehot.call(target).to(device=device,dtype=torch.long)

        optimizer.zero_grad()
        prediction = net(source) # # (N,4,256,256)
        # print(target_)
        # print("\n")
        # print(prediction)
        # loss = criterion(prediction, target_) 
        loss =  criterion(prediction, target_)  + dice_loss(F.softmax(prediction, dim=1).float(),
                                       F.one_hot(target_, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(epoch, epoch_loss)
        # print(target, target_)
        # if target[target!=0].size()[0]>0:
            # np.save("./target.npy", target.cpu().numpy())
            # np.save("./target_.npy", target_.cpu().numpy())
            # a = target.cpu().numpy()[0]
            # b= target_.cpu().numpy()[0]
            # mask = a[1,:,:]==221
            # if np.any(mask):
                # print(221,b[:,mask][:,0])
            # mask = a[1,:,:]==255
            # if np.any(mask):
                # print(255,b[:,mask][:,0])
            # mask = a[1,:,:]==50
            # if np.any(mask):
        
                # print(50,b[:,mask][:,0])
            
            # assert 1==0

            # print(epoch, i_batch, prediction[0,:,0,0], target_[target_!=0])
        # print(i_batch, sample_batched['img'].size(),
            #   sample_batched['label'].size(), sample_batched['img'].mean(), sample_batched['label'][sample_batched['label']!=0], prediction.shape, prediction.mean())
        # break
    # break