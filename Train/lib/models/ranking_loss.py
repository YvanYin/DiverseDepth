import torch
from torch import nn


def randomSampling(inputs, targets, masks, threshold, sample_num):

    # find A-B point pairs from predictions
    inputs_index = torch.masked_select(inputs, targets.gt(threshold))
    num_effect_pixels = len(inputs_index)
    shuffle_effect_pixels = torch.randperm(num_effect_pixels).cuda()
    inputs_A = inputs_index[shuffle_effect_pixels[0:sample_num*2:2]]
    inputs_B = inputs_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # find corresponding pairs from GT
    target_index = torch.masked_select(targets, targets.gt(threshold))
    targets_A = target_index[shuffle_effect_pixels[0:sample_num*2:2]]
    targets_B = target_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # only compute the losses of point pairs with valid GT
    consistent_masks_index = torch.masked_select(masks, targets.gt(threshold))
    consistent_masks_A = consistent_masks_index[shuffle_effect_pixels[0:sample_num*2:2]]
    consistent_masks_B = consistent_masks_index[shuffle_effect_pixels[1:sample_num*2:2]]

    # The amount of A and B should be the same!!
    if len(targets_A) > len(targets_B):
        targets_A = targets_A[:-1]
        inputs_A = inputs_A[:-1]
        consistent_masks_A = consistent_masks_A[:-1]

    return inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B


def ind2sub(idx, cols):
    r = idx / cols
    c = idx - r * cols
    return r, c

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx

######################################################
# Ranking loss (Random sampling)
#####################################################
class RankingLoss(nn.Module):
    def __init__(self, point_pairs=5000, sigma=0.03, alpha=1.0, mask_value=1e-8):
        super(RankingLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.sigma = sigma # used for determining the ordinal relationship between a selected pair
        self.alpha = alpha  # used for balancing the effect of = and (<,>)
        self.mask_value = mask_value

    def forward(self, inputs, targets, masks=None):
        n,c,h,w = targets.size()
        if masks == None:
            masks = targets > self.mask_value
        if n != 1:
            inputs = inputs.view(n, -1).double()
            targets = targets.view(n, -1).double()
            masks = masks.view(n, -1).double()
        else:
            inputs = inputs.contiguous().view(1, -1).double()
            targets = targets.contiguous().view(1, -1).double()
            masks = masks.contiguous().view(1, -1).double()

        loss = torch.DoubleTensor([0.]).cuda()
        for i in range(n):
            # find A-B point pairs
            inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B = randomSampling(inputs[i,:], targets[i, :], masks[i, :], self.mask_value, self.point_pairs)

            #GT ordinal relationship
            target_ratio = torch.div(targets_A, targets_B+1e-8)
            mask_eq = target_ratio.lt(1.0 + self.sigma) * target_ratio.gt(1.0/(1.0+self.sigma))
            labels = torch.zeros_like(target_ratio)
            labels[target_ratio.ge(1.0 + self.sigma)] = 1
            labels[target_ratio.le(1.0/(1.0+self.sigma))] = -1

            # consider forward-backward consistency checking, only compute the losses of point pairs with valid GT
            consistency_mask = consistent_masks_A * consistent_masks_B

            # compute loss
            equal_loss = (inputs_A - inputs_B).pow(2) * mask_eq.double() * consistency_mask
            unequal_loss = torch.log(1 + torch.exp((-inputs_A + inputs_B) * labels)) * (~mask_eq).double() * consistency_mask

            loss = loss + self.alpha * torch.sum(equal_loss)/(equal_loss.numel() + 1e-8) + torch.sum(unequal_loss) / (unequal_loss.numel() + 1e-8)

        return loss[0].float()/(n + 1e-8)

