import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# TODO
# Implement Marginal Loss

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6, ignore_channel=None):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6, ignore_channel: int = 0):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        if channel==ignore_channel:
            # print("ignore channel ", channel)
            
            continue
        # print("channel ", channel, target[0, :, 0,:])
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False, ignore_channel: int = 0):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True, ignore_channel=ignore_channel)

class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes=4, multiclass=True):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.multiclass = multiclass
        # print("self.n_classes self.multiclass: ",  self.n_classes, self.multiclass)
    def forward(self, prediction, target, ignore_channel=None):
        print("self.n_classes self.multiclass: ",  self.n_classes, self.multiclass, torch.unique(target))
        
        return dice_loss(F.softmax(prediction, dim=1).float(),
                                       F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=self.multiclass, ignore_channel=ignore_channel)
def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def merged_onehot_transform(tensor, depth=4, data_type="float", new_n_classes=2, index=2):
    assert new_n_classes==2
    assert index==2
    assert depth==4
    onehot = F.one_hot(tensor, depth).permute(0, 3, 1, 2) # (bs, c, h, w), c=depth
    bs,c,h,w = onehot.shape
    onehot = onehot.permute(1, 0, 2, 3)# (bs, c, h, w)->( c, bs, h, w)
    # (new_n_classes, bs, h, w)
    merged_onehot = torch.zeros((new_n_classes, bs, h ,w))
    mask = onehot[index]==1 # 0010 -> 01; others 10
    merged_onehot[1, mask]=1
    merged_onehot[0, ~mask]=1
    merged_onehot = merged_onehot.permute(1, 0, 2, 3) # -> (new_n_classes, bs, h, w)->(bs, new_n_classes, h, w)
    
    if data_type=="float":
        
        return merged_onehot.float()
    else:
        raise NotImplementedError("nah son")
    
### seems to have a bug when target has 0,1,2,3
def expand_gt_squeezed(net_output, tensor, cur_task, default_task, depth=4, index=2, new_n_classes=2):
    assert index==2
    assert new_n_classes==2
    assert depth==4
    target_onehot = F.one_hot(tensor, depth).permute(0, 3, 1, 2) # (bs, c, h, w), c=depth
    
    target_onehot = target_onehot.permute(1, 0, 2, 3)# (bs, c, h, w)->( c, bs, h, w)

    target_onehot = target_onehot.view(depth, -1)# ->( c, bsxhxw)
    
    new_gt = torch.zeros((new_n_classes, target_onehot.shape[1]))
    if net_output.device.type == "cuda":
        new_gt = new_gt.cuda(net_output.device.index)
    mask = target_onehot[index]==1
    
    for i, task in enumerate(default_task):
        
        if task in cur_task: 
            # print(i,task, "cur_task")
            j = cur_task.index(task) # i=2, j=0
            
            new_gt[1, mask]+=target_onehot[i, mask]
        else:
            # print(i,task, "not cur_task")
            
            new_gt[0, ~mask]+=target_onehot[i, ~mask]
    
    return new_gt

def merge_prediction_squeezed(net_output, target_onehot, cur_task, default_task, depth=4, index=2, new_n_classes=2):
    assert index==2
    assert new_n_classes==2
    assert depth==4
    # target_onehot (new_n_classes, -1)
    # net_output (bs, c, h, w)
    net_output = net_output.permute(1, 0, 2, 3)# (bs, c, h, w)->( c, bs, h, w)
    # print(net_output.shape)
    net_output = net_output.contiguous().view(depth, -1)# ->( c, bsxhxw)
    net_output_onehot = F.softmax(net_output, dim=0)#( c, bsxhxw)
    new_prediction = torch.zeros((new_n_classes, target_onehot.shape[1]))
    if net_output.device.type == "cuda":
        new_prediction = new_prediction.cuda(net_output.device.index)
    mask = torch.argmax(net_output_onehot,dim=0)==index #( bsxhxw)
    
    for i, task in enumerate(default_task):
        
        if task in cur_task: 
            # print(i,task, "cur_task")
            j = cur_task.index(task) # i=2, j=0
            
            new_prediction[1, mask]+=net_output[i, mask]
        else:
            # print(i,task, "not cur_task")
            
            new_prediction[0, ~mask]+=net_output[i, ~mask]
    
    return new_prediction

def merged_squeezed_onehot_transform(tensor, depth=4, data_type="float", new_n_classes=2, index=2):
    assert new_n_classes==2
    assert index==2
    assert depth==4
    onehot = F.one_hot(tensor, depth).permute(0, 3, 1, 2) # (bs, c, h, w), c=depth
    
    onehot = onehot.permute(1, 0, 2, 3)# (bs, c, h, w)->( c, bs, h, w)

    onehot = onehot.view(depth, -1)# ->( c, bsxhxw)
    
    # # (c, bsxhxw) -> (new_n_classes, bsxhxw)
    merged_onehot = torch.zeros((new_n_classes, onehot.shape[1]))
    mask = onehot[index,:]==1 # 0010 -> 01; others 10
    merged_onehot[1, mask]=1
    merged_onehot[0, ~mask]=1
    
    
    if data_type=="float":
        
        return merged_onehot.float() # (new_n_classes, bsxhxw)
    else:
        raise NotImplementedError("nah son")
    

def onehot_transform(tensor, depth, data_type="float", squeeze=False):
    onehot = F.one_hot(tensor, depth).permute(0, 3, 1, 2) # (bs, c, h, w), c=depth
    if squeeze:
        onehot = onehot.permute(1, 0, 2, 3)# (bs, c, h, w)->( c, bs, h, w)
        
        onehot = onehot.view(depth, -1)
    if data_type=="float":
        
        return onehot.float()
    else:
        raise NotImplementedError("nah son")
    


class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target, ignore_bk=False):
        # print(inp.shape) # torch.Size([2, 4, 512, 512])
        # print(target.dtype) # torch.int64
        
        # target = target.long()
        num_classes = inp.size()[1]
        # print("num_classes ", num_classes) # 4
        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)
        # print("inp.shape ",inp.shape) #(N,4)
        # print("target.shape ",target.shape) ##(N,) # here we can apply a mask
        # print(target.shape, target.max(),target.min())
        if ignore_bk:
            # print("ignore_bk ", ignore_bk)
            valid_mask = (target!=0) #(N,)

            return super(CrossentropyND, self).forward(inp*valid_mask.view(-1,1), target*valid_mask)
        else:
            return super(CrossentropyND, self).forward(inp, target)
            





class DC_CE_Marginal_Exclusion_loss(nn.Module):
    def __init__(self, n_classes=4, multiclass=True, aggregate="sum", ex=False, new_n_classes=2, index=2, h=512,w=512):
        super(DC_CE_Marginal_Exclusion_loss, self).__init__()
        self.aggregate = aggregate
        self.n_classes = n_classes
        self.ex_choice = ex
        
        self.new_n_classes=new_n_classes
        self.index=index
        self.h=h
        self.w=w
        
        self.ce = CrossentropyND()
        self.dc = DiceLoss(n_classes=self.n_classes, multiclass=multiclass)
        self.dc_m = DiceLoss(n_classes=self.new_n_classes, multiclass=False)
        
        # self.ex = Exclusion_loss(self.dc)
        # self.ex_CE = Exclusion_loss(self.ce)

        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, default_task, cur_task):
        # if cur_task!=default_task: meaning you can trust the label from cur_task to be fully annotated (in our case soil)
        if cur_task == default_task:
            # print(f"equal:{cur_task}/{default_task}")
            # dc_loss = self.dc(net_output, target)
            # ce_loss = self.ce(net_output, target)
            
            # normal loss without background
            dc_loss = self.dc(net_output, target,ignore_channel=0)
            ce_loss = self.ce(net_output, target, ignore_bk=True)
            if self.aggregate == "sum":
                result = ce_loss + dc_loss
            elif self.aggregate == "ce":
                result = ce_loss
            elif self.aggregate == "dc":
                result = dc_loss
            else:
                # reserved for other stuff (later?)
                raise NotImplementedError("nah son")
        # if self.ex_choice:
            # adding marginal loss
            # pass
    
        else:
            
            # print(f"not_equal:{cur_task}/{default_task}")
            # target_onehot = merged_onehot_transform(target)
            assert torch.unique(target.long()).size()[0]<=4
            target_onehot_merged_squeezed = merged_squeezed_onehot_transform(target, depth=self.n_classes, data_type="float", new_n_classes=self.new_n_classes, index=self.index) #(new_n_classes, bsxhxw)
            
            # target_onehot_merged_squeezed = expand_gt_squeezed(net_output, target, cur_task, default_task, depth=self.n_classes, index=self.index, new_n_classes=self.new_n_classes) #(2,-1)
            merged_pre = merge_prediction_squeezed(net_output, target_onehot_merged_squeezed, cur_task, default_task, depth=self.n_classes, index=self.index, new_n_classes=self.new_n_classes)#(2,-1)
            # target_onehot = onehot_transform(target, self.n_classes, data_type="float") # (bs, H, W) -> (bs, self.n_classes, H, W)
            target_onehot_merged_squeezed = target_onehot_merged_squeezed.view(-1,merged_pre.shape[0],self.h, self.w)
            merged_pre = merged_pre.view(-1,merged_pre.shape[0],self.h,self.w)
            target_merged_squeezed = torch.argmax(target_onehot_merged_squeezed, axis=0)
            if torch.unique(target_merged_squeezed).size()[0]>self.new_n_classes:
                print("error: saving for debugging")
                import numpy as np
                np.save("./target_before_onehot.npy", target.cpu().numpy())
                np.save("./target_onehot_merged_squeezed.npy", target_onehot_merged_squeezed.cpu().numpy())
                np.save("./merged_pre.npy", merged_pre.detach().cpu().numpy())
                np.save("./target_merged_squeezed.npy", target_merged_squeezed.cpu().numpy())
                print("torch.unique(target_merged_squeezed) ", torch.unique(target_merged_squeezed))
                
            assert torch.unique(target_merged_squeezed).size()[0]<self.new_n_classes+1
            dc_loss_m = self.dc_m(merged_pre, target_merged_squeezed)
            ce_loss_m = self.ce(merged_pre, target_merged_squeezed)
            if self.aggregate == "sum":
                result = ce_loss_m + dc_loss_m
            elif self.aggregate == "ce":
                result = ce_loss_m
            elif self.aggregate == "dc":
                result = dc_loss_m
            else:
                # reserved for other stuff (later?)
                raise NotImplementedError("nah son")

        return result
            # target_onehot1 = onehot_transform(target, self.n_classes, data_type="float", squeeze=True) # (bs, H, W) -> (bs, self.n_classesxHxW)
            
            # print("target_onehot, target, net_output ", target_onehot.shape, target.shape, net_output.shape) # torch.Size([2, 4, 512]), torch.Size([2, 512, 512]), torch.Size([2, 4, 512, 512])
            
            # print("target_onehot1.shape ",target_onehot1.shape)
            # import numpy as np
            # np.save("./target_before_onehot.npy", target.cpu().numpy())
            # np.save("./target_onehot_merged_squeezed.npy", target_onehot_merged_squeezed.cpu().numpy())
            # np.save("./target_onehot_merged_squeezed1.npy", target_onehot_merged_squeezed1.cpu().numpy())
            
            # np.save("./target_onehot_ori.npy", target_onehot_ori.cpu().numpy())
            # not_gt = expand_gt_new_view(net_output, target_onehot,
                               # cur_task, default_task) # (bs,c,h,w) merged
            
            # TO DO DEBUG
            # merged_pre = merge_prediction_new_view(
                # net_output, target_onehot, cur_task, default_task)
            # merged_pre = merge_prediction_new_view(
                # target_onehot_ori, target_onehot, cur_task, default_task)
            
            
            # print("not_gt.shape ",not_gt.shape)
            # np.save("./not_gt.npy", not_gt.cpu().numpy())
            # np.save("./merged_pre.npy", merged_pre.detach().cpu().numpy())
            # np.save("./net_output.npy", net_output.detach().cpu().numpy())
            # np.save("./net_output.npy", target_onehot_ori.cpu().numpy())
            
            
            
            # merged_pre = merge_prediction(
                # net_output, target_onehot1, cur_task, default_task)
            # assert 1==0
#             merged_pre = merge_prediction(
#                 net_output, target_onehot, cur_task, default_task)
#             not_gt = expand_gt(net_output, target_onehot,
#                                cur_task, default_task)
#             dc_loss = self.dc(merged_pre, target)
#             ce_loss = self.ce(merged_pre, target)
#             ex_loss = self.ex(net_output, not_gt)
#             # epsilon=1
#             # ex_loss = self.ex(net_output, not_gt)+self.ex_CE(net_output+epsilon,not_gt)
#             if self.aggregate == "sum":
#                 result = ce_loss + dc_loss
#             elif self.aggregate == "ce":
#                 result = ce_loss
#             elif self.aggregate == "dc":
#                 result = dc_loss
#             else:
#                 # reserved for other stuff (later?)
#                 raise NotImplementedError("nah son")
#             if self.ex_choice:
#                 result = result+2*ex_loss
#                 # result = ex_loss
        # return result
