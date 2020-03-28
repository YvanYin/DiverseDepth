from . import lateral_net as network
from lib.utils.net_tools import *
from lib.models.image_transfer import *
from lib.core.config import cfg
import torch
import torch.nn.functional


class RelDepthModel(nn.Module):
    def __init__(self):
        super(RelDepthModel, self).__init__()
        self.loss_names = ['Virtual_Normal']
        self.depth_model = DepthModel()

    def forward(self, data):
        # Input data is a_real, predicted data is b_fake, groundtruth is b_real
        self.a_real = data['A'].cuda()
        self.b_fake_logit, self.b_fake_softmax = self.depth_model(self.a_real)
        return {'b_fake_logit': self.b_fake_logit, 'b_fake_softmax': self.b_fake_softmax}

    def inference(self, data):
        with torch.no_grad():
            out = self.forward(data)

            if cfg.MODEL.PREDICTION_METHOD == 'classification':
                pred_depth = bins_to_depth(out['b_fake_softmax'])
            elif cfg.MODEL.PREDICTION_METHOD == 'regression':
                # for regression methods
                #pred_depth = torch.nn.functional.sigmoid(out['b_fake_logit'])
                pred_depth = out['b_fake_logit']
                pred_depth = torch.abs(pred_depth - pred_depth.min() + 1)  #pred_depth - pred_depth.min() #- pred_depth.max()
            else:
                raise ValueError("Unknown prediction methods")

            out = pred_depth
            return {'b_fake': out}

    def inference_kitti(self, data):
        # crop kitti images into 3 parts0.401
        with torch.no_grad():
            self.a_l_real = data['A_l'].cuda()
            b_l_logit, b_l_classes = self.depth_model(self.a_l_real)

            self.a_m_real = data['A_m'].cuda()
            b_m_logit, b_m_classes = self.depth_model(self.a_m_real)

            self.a_r_real = data['A_r'].cuda()
            b_r_logit, b_r_classes = self.depth_model(self.a_r_real)
            if cfg.MODEL.PREDICTION_METHOD == 'classification':
                self.b_l_fake = bins_to_depth(b_l_classes)
                self.b_m_fake = bins_to_depth(b_m_classes)
                self.b_r_fake = bins_to_depth(b_r_classes)
            elif cfg.MODEL.PREDICTION_METHOD == 'regression':
                self.b_l_fake = torch.abs(b_l_logit - b_l_logit.min() + 1)
                self.b_m_fake = torch.abs(b_m_logit - b_m_logit.min() + 1)
                self.b_r_fake = torch.abs(b_r_logit - b_r_logit.min() + 1)
            out = kitti_merge_imgs(self.b_l_fake, self.b_m_fake, self.b_r_fake, torch.squeeze(data['B_raw']).shape,
                                   data['crop_lmr'])
            return {'b_fake': out}


class ModelLoss(object):
    def __init__(self):
        super(ModelLoss, self).__init__()
        self.virtual_normal_loss = VNL_Loss(focal_x=cfg.DATASET.FOCAL_X, focal_y=cfg.DATASET.FOCAL_Y,
                                            input_size=cfg.DATASET.CROP_SIZE, sample_ratio=0.15)
        self.ranking_loss = Ranking_Loss(sample_ratio=0.08)
        self.weight_cross_entropy_loss = WCEL_Loss()
        self.scale_shift_invariant_loss = SSIL_Loss()
        self.multi_scale_gradient_loss = MSGL_Loss(scale=1, scale_shift_depth=True)
        self.youtube3d_ranking_loss = YouTube3D_Ranking_Loss()

    def criterion(self, pred_softmax, pred_logit, data):
        if cfg.MODEL.PREDICTION_METHOD == 'classification':
            pred_depth = bins_to_depth(pred_softmax)
        elif cfg.MODEL.PREDICTION_METHOD == 'regression':
            pred_depth = pred_logit
        else:
            raise ValueError("Unknown prediction methods")

        gt_depth = data['B'].to(device=pred_depth.device)

        loss = {}
        if 'youtube3dordinal' in cfg.TRAIN.LOSS_MODE.lower():
            loss['youtube3d-ranking_loss'] = self.youtube3d_ranking_loss(pred_depth, gt_depth, data)
        if 'ranking' in cfg.TRAIN.LOSS_MODE.lower():
            loss['ranking_loss'] = self.ranking_loss(pred_depth, gt_depth)
        if 'vnl' in cfg.TRAIN.LOSS_MODE.lower():
            loss['virtual_normal_loss'] = self.virtual_normal_loss(gt_depth, pred_depth)
        if 'ssil' in cfg.TRAIN.LOSS_MODE.lower():
            loss_ssi = self.scale_shift_invariant_loss(pred_depth, gt_depth)
            loss['ssi_loss'] = loss_ssi
        if 'msgl' in cfg.TRAIN.LOSS_MODE.lower():
            loss['multi-scale_gradient_loss'] = self.multi_scale_gradient_loss(pred_depth, gt_depth)
        if 'sdepth' in cfg.TRAIN.LOSS_MODE.lower():
            mask = data['mask_samples']
            gt_depth = gt_depth[mask]
            pred_depth = pred_depth[mask]
            if torch.numel(gt_depth) > 0:
                loss['sample-depth_mse_loss'] = torch.nn.functional.mse_loss(pred_depth, gt_depth)
            else:
                loss['sample-depth_mse_loss'] = 0

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
        #bottom_up_model = 'lateral_net.lateral_' + cfg.MODEL.ENCODER
        bottom_up_model = network.__name__.split('.')[-1] + '.lateral_' + cfg.MODEL.ENCODER
        self.encoder_modules = get_func(bottom_up_model)()
        #self.decoder_modules = lateral_net.fcn_topdown(cfg.MODEL.ENCODER)
        self.decoder_modules = network.fcn_topdown()

    def forward(self, x):
        lateral_out, encoder_stage_size = self.encoder_modules(x)
        out_logit, out_softmax = self.decoder_modules(lateral_out, encoder_stage_size)
        return out_logit, out_softmax


def cal_params(model):
    model_dict = model.state_dict()
    paras = np.sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum = 0

    for key in model_dict.keys():
        print(key)
        if 'layer5' not in key:
            if 'running' not in key:
                print(key)
                ss = model_dict[key].size()
                temp = 1
                for s in ss:
                    temp = temp * s
                print(temp)
                sum = sum + temp
    print(sum)
    print(paras)

