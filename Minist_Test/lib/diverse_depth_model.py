import Minist_Test.lib.lateral_net  as network
from Minist_Test.lib.net_tools import *
import torch
import torch.nn.functional
import torch.nn as nn


class RelDepthModel(nn.Module):
    def __init__(self):
        super(RelDepthModel, self).__init__()
        self.depth_model = DepthModel()

    def forward(self, data):
        # Input data is a_real, predicted data is b_fake, groundtruth is b_real
        input = data.cuda()
        predict, _ = self.depth_model(input)
        return predict

    def inference(self, data):
        with torch.no_grad():
            out = self.forward(data)
            pred_depth = torch.abs(out - out.min() + 1)  #pred_depth - pred_depth.min() #- pred_depth.max()
            return pred_depth

class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        self.encoder_modules = network.lateral_resnext50_32x4d_body_stride16()
        self.decoder_modules = network.fcn_topdown()

    def forward(self, x):
        lateral_out, encoder_stage_size = self.encoder_modules(x)
        out_logit, out_softmax = self.decoder_modules(lateral_out, encoder_stage_size)
        return out_logit, out_softmax
