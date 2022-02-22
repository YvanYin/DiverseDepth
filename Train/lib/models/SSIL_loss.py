import torch
import torch.nn as nn


class SSIL_Loss(nn.Module):
    """
    Scale and Shift Invariant Loss
    """
    def __init__(self, valid_threshold=1e-8, max_threshold=1e8):
        super(SSIL_Loss, self).__init__()
        self.valid_threshold = valid_threshold
        self.max_threshold = max_threshold

    def inverse(self, mat):
        bt, _, _ = mat.shape
        mat += 1e-7 * torch.eye(2, dtype=mat.dtype, device=mat.device)
        a = mat[:, 0, 0]
        b = mat[:, 0, 1]
        c = mat[:, 1, 0]
        d = mat[:, 1, 1]
        ad_bc = a * d - b * c
        out = torch.zeros_like(mat)
        mat = mat / ad_bc[:, None, None]
        out[:, 0, 0] = mat[:, 1, 1]
        out[:, 0, 1] = -mat[:, 0, 1]
        out[:, 1, 0] = -mat[:, 1, 0]
        out[:, 1, 1] = mat[:, 0, 0]
        return out

    def scale_pred_depth_mask(self, pred, gt, logger=None):
        b, c, h, w = pred.shape
        mask = (gt > self.valid_threshold)  & (gt < self.max_threshold)  # [b, c, h, w]
        mask_float = mask.to(dtype=pred.dtype, device=pred.device)
        pred_valid = pred * mask_float   # [b, c, h, w]
        ones_valid_shape = torch.ones_like(pred_valid) * mask_float  # [b, c, h, w]
        pred_valid_ones = torch.cat((pred_valid, ones_valid_shape), dim=1)  # [b, c+1, h, w]
        pred_valid_ones_reshape = pred_valid_ones.reshape((b, c + 1, -1))  # [b, c+1, h*w]

        A = torch.matmul(pred_valid_ones_reshape, pred_valid_ones_reshape.permute(0, 2, 1))  # [b, 2, 2]

        # print(A)
        #A_inverse = (A + 1e-7 * torch.eye(2, dtype=A.dtype, device=A.device)).inverse() # this may get identity matrix in some versions of Pytorch. If it occurs, add 'print(A)' before it can solve it
        A_inverse = self.inverse(A)
        gt_valid = gt * mask_float
        gt_reshape = gt_valid.reshape((b, c, -1))  # [b, c, h*w]
        B = torch.matmul(pred_valid_ones_reshape, gt_reshape.permute(0, 2, 1))  # [b, 2, 1]
        scale_shift = torch.matmul(A_inverse, B)  # [b, 2, 1]
        ones = torch.ones_like(pred)
        pred_ones = torch.cat((pred, ones), dim=1)  # [b, 2, h, w]
        pred_scale_shift = torch.matmul(pred_ones.permute(0, 2, 3, 1).reshape(b, h * w, 2), scale_shift)  # [b, h*w, 1]
        pred_scale_shift = pred_scale_shift.permute(0, 2, 1).reshape((b, c, h, w))

        return pred_scale_shift, mask

    def forward(self, pred, gt, logger=None):
        pred_scale, mask_valid = self.scale_pred_depth_mask(pred, gt)
        valid_pixs = torch.sum(mask_valid, (1, 2, 3))
        valid_batch = valid_pixs > 50
        diff = torch.abs(pred_scale * valid_batch[:, None, None, None] * mask_valid -
                         gt*valid_batch[:, None, None, None]*mask_valid)
        loss = torch.sum(diff) / (torch.sum(valid_batch[:, None, None, None] * mask_valid) + 1e-8)
        return loss



if __name__ == '__main__':
    pred = torch.rand(3, 1, 100, 100).cuda()
    gt = torch.rand(3, 1, 100, 100).cuda()
    SSIL = SSIL_Loss()
    loss = SSIL(pred, gt)
    print(loss)

