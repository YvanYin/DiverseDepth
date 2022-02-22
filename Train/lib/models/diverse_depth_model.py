from . import lateral_net as network
from lib.utils.net_tools import *
from lib.configs.config import cfg
import torch
import torch.nn.functional
from lib.models.VNL_rel_loss import VNL_Loss
from lib.models.ranking_loss import RankingLoss
from lib.models.SSIL_loss import SSIL_Loss

class RelDepthModel(nn.Module):
    def __init__(self):
        super(RelDepthModel, self).__init__()
        self.depth_model = DepthModel()
        self.losses = ModelLoss()

    def forward(self, data, logger=None):
        # Input data is a_real, predicted data is b_fake, groundtruth is b_real
        inputs = data['A'].cuda()
        pred = self.depth_model(inputs)
        losses_dict = self.losses.criterion(pred, data)
        return {'pred': pred, 'losses': losses_dict}

    def inference(self, data):
        with torch.no_grad():
            inputs = data['A'].cuda()
            pred_logit = self.depth_model(inputs)
            # for regression methods
            #pred_depth = torch.nn.functional.sigmoid(out['b_fake_logit'])
            pred_depth = torch.abs(pred_logit - pred_logit.min() + 1)  #pred_depth - pred_depth.min() #- pred_depth.max()
            return {'pred': pred_depth}

class ModelLoss(object):
    def __init__(self):
        super(ModelLoss, self).__init__()
        self.virtual_normal_loss = VNL_Loss(focal_x=cfg.DATASET.FOCAL_X, focal_y=cfg.DATASET.FOCAL_Y,
                                            input_size=cfg.DATASET.CROP_SIZE, sample_ratio=0.15)
        self.ranking_loss = RankingLoss()
        self.scale_shift_invariant_loss = SSIL_Loss()
        #self.multi_scale_gradient_loss = MSGL_Loss(scale=1, scale_shift_depth=True)

    def criterion(self, pred_logit, data, logger=None):
        pred_depth = pred_logit

        gt_depth = data['B'].to(device=pred_depth.device)

        loss = {}
        if '_ranking_' in cfg.TRAIN.LOSS_MODE.lower():
            loss['ranking_loss'] = self.ranking_loss(pred_depth, gt_depth)
        if '_vnl_' in cfg.TRAIN.LOSS_MODE.lower():
            loss['virtual_normal_loss'] = self.virtual_normal_loss(gt_depth, pred_depth)
        if '_ssil_' in cfg.TRAIN.LOSS_MODE.lower():
            loss_ssi = self.scale_shift_invariant_loss(pred_depth, gt_depth)
            loss['ssi_loss'] = loss_ssi
        #if 'msgl' in cfg.TRAIN.LOSS_MODE.lower():
        #    loss['multi-scale_gradient_loss'] = self.multi_scale_gradient_loss(pred_depth, gt_depth)

        total_loss = sum(loss.values())
        loss['total_loss'] = total_loss
        return loss


class ModelOptimizer(object):
    def __init__(self, model):
        super(ModelOptimizer, self).__init__()
        encoder_params = []
        encoder_params_names = []
        decoder_params = []
        decoder_params_names = []
        nograd_param_names = []

        for key, value in model.named_parameters():
            if value.requires_grad:
                if 'res' in key:
                    encoder_params.append(value)
                    encoder_params_names.append(key)
                else:
                    decoder_params.append(value)
                    decoder_params_names.append(key)
            else:
                nograd_param_names.append(key)

        lr_encoder = cfg.TRAIN.BASE_LR
        lr_decoder = cfg.TRAIN.BASE_LR * cfg.TRAIN.SCALE_DECODER_LR
        weight_decay = 0.0005

        net_params = [
            {'params': encoder_params,
             'lr': lr_encoder,
             'weight_decay': weight_decay},
            {'params': decoder_params,
             'lr': lr_decoder,
             'weight_decay': weight_decay},
        ]
        self.optimizer = torch.optim.SGD(net_params, momentum=0.9)

    def optim(self, loss):
        self.optimizer.zero_grad()
        loss_all = loss['total_loss']
        loss_all.backward()
        self.optimizer.step()


class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        bottom_up_model = network.__name__.split('.')[-1] + '.lateral_' + cfg.MODEL.ENCODER
        self.encoder_modules = get_func(bottom_up_model)()
        self.decoder_modules = network.fcn_topdown()

    def forward(self, x):
        lateral_out, encoder_stage_size = self.encoder_modules(x)
        out_logit, _ = self.decoder_modules(lateral_out, encoder_stage_size)
        return out_logit
