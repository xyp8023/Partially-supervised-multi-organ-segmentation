import torch
# from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND
from partial.utils.ND_Crossentropy import CrossentropyND
# from nnunet.training.loss_functions.TopK_loss import TopKLoss
# from nnunet.utilities.nd_softmax import softmax_helper
# from nnunet.utilities.tensor_utilities import sum_tensor
# from nnunet.utilities.mk_utils import get_tag_index
from torch import nn
import numpy as np

class DC_CE_Marginal_Exclusion_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="ce", ex=True):
        super(DC_CE_Marginal_Exclusion_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        # self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        # self.ex = Exclusion_loss(self.dc)
        # self.ex_CE = Exclusion_loss(self.ce)
        # self.ex_choice = ex
        # print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, default_task, cur_task):
        if cur_task == default_task:
            # print(f"equal:{cur_task}/{default_task}")
            # dc_loss = self.dc(net_output, target)
            ce_loss = self.ce(net_output, target)
            result = ce_loss

            # if self.aggregate == "sum":
                # result = ce_loss + dc_loss
            # elif self.aggregate == "ce":
                # result = ce_loss
            # elif self.aggregate == "dc":
                # result = dc_loss
            # else:
                # reserved for other stuff (later?)
                # raise NotImplementedError("nah son")
        else:
            # print(f"not_equal:{cur_task}/{default_task}")
            target_onehot = onehot_transform(target, len(cur_task)+1)
            merged_pre = merge_prediction(
                net_output, target_onehot, cur_task, default_task)
            not_gt = expand_gt(net_output, target_onehot,
                               cur_task, default_task)
            dc_loss = self.dc(merged_pre, target)
            ce_loss = self.ce(merged_pre, target)
            ex_loss = self.ex(net_output, not_gt)
            # epsilon=1
            # ex_loss = self.ex(net_output, not_gt)+self.ex_CE(net_output+epsilon,not_gt)
            if self.aggregate == "sum":
                result = ce_loss + dc_loss
            elif self.aggregate == "ce":
                result = ce_loss
            elif self.aggregate == "dc":
                result = dc_loss
            else:
                # reserved for other stuff (later?)
                raise NotImplementedError("nah son")
            if self.ex_choice:
                result = result+2*ex_loss
                # result = ex_loss
        return result

