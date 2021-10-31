import torch
import numpy as np
from dataloading.dataloading import WeedCropDataset, ToTensor, RandomCrop
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from unet import UNet
from utils.dice_score import *
from utils.utils import To_Onehot, To_Class

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import sys

## hyper-parameters
learning_rate=0.002
amp=False
# torch.manual_seed(17)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transforms = transforms.Compose([transforms.ToTensor()]) 
# train_transforms = transforms.Compose([transforms.RandomCrop(512), transforms.ToTensor()]) 



dataset = WeedCropDataset(root_dir="datasets/debug", transform=train_transforms, img_file="img_name.txt", label_file="label_name.txt")
# dataset = WeedCropDataset(root_dir="datasets/val")
N = 2 # batch_size
dataloader = DataLoader(dataset, batch_size=N,
                        shuffle=True, num_workers=2)

net = UNet(n_channels=3, n_classes=4, bilinear=True)
net.to(device=device)
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
# grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
# criterion = nn.CrossEntropyLoss()
# criterion = CrossentropyND()
# dc = DiceLoss(n_classes=4, multiclass=True)
# soft_dice_loss = SoftDiceLoss(apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1.)

criterion = DC_CE_Marginal_Exclusion_loss(n_classes=4, multiclass=True)


# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
debug=False
num_epochs= 2 if debug else 2001
    
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        source = sample_batched['img'].to(device=device, dtype=torch.float32) # (N,3,256,256)
        # target = (sample_batched['label']*255).to(device=device, dtype=torch.uint8)# (N,3,256,256)
        target = sample_batched['label'].to(device=device)# (N,256,256)
        # print(epoch, source.mean())

        optimizer.zero_grad()
        prediction = net(source) # # (N,4,256,256)
        # if target[target!=0].size()[0]>0:
            # np.save("./target.npy", target.cpu().numpy())
            # np.save("./target1.npy", (sample_batched['label']*255))
            # assert 1==0

        # loss =  criterion(prediction, target)  + dice_loss(F.softmax(prediction, dim=1).float(),
                                       # F.one_hot(target, net.n_classes).permute(0, 3, 1, 2).float(),
                                       # multiclass=True)
        # loss = criterion(prediction, target) + dc(prediction, target)
        loss = criterion(prediction, target, default_task=["b", "c", "s", "w"], cur_task=["b", "c", "s", "w"]) + criterion(prediction, target, default_task=["b", "c", "s", "w"], cur_task=["s"])
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(epoch, epoch_loss)

    
PATH = "checkpoints"+os.sep+"epoch_"+str(epoch)+"-loss_"+str(epoch_loss)+".pt"
if sys.argv[1]=='save':
    torch.save(net.state_dict(), PATH)
