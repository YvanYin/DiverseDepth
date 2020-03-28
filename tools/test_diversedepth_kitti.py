from tools.parse_arg_test import TestOptions
from data.load_dataset import CustomerDataLoader
from lib.models.diverse_depth_model import RelDepthModel
from lib.utils.net_tools import load_ckpt
import torch
import torch.nn
from lib.models.image_transfer import resize_image
from lib.utils.evaluate_depth_error import evaluate_rel_err, recover_metric_depth
import os
import cv2
import numpy as np
from lib.core.config import cfg, merge_cfg_from_file
import matplotlib.pyplot as plt
from lib.utils.logging import setup_logging, SmoothedValue
from torchvision.transforms import transforms
logger = setup_logging(__name__)


def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    img = np.transpose(img, (2, 0, 1))
    img = img[::-1, :, :]
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
    return img

def test_kitti_split(A, depth, model):
    A_l = A[:, :, :, :int(A.size(3) / 3) + 5]
    A_m = A[:, :, :, int(A.size(3) / 3) - 5:int(A.size(3) / 3) * 2 + 5]
    A_r = A[:, :, :, int(A.size(3) / 3) * 2 - 5:]

    pred_depth_l, _ = model.module.depth_model(A_l)
    pred_depth_m, _ = model.module.depth_model(A_m)
    pred_depth_r, _ = model.module.depth_model(A_r)

    pred_depth_metric_l = recover_metric_depth(pred_depth_l, depth[:, :, :int(A.size(3) / 3) + 5])
    pred_depth_metric_m = recover_metric_depth(pred_depth_m,
                                               depth[:, :, int(A.size(3) / 3) - 5:int(A.size(3) / 3) * 2 + 5])
    pred_depth_metric_r = recover_metric_depth(pred_depth_r, depth[:, :, int(A.size(3) / 3) * 2 - 5:])
    pred_depth_metric = np.zeros((A.size(2), A.size(3)))
    pred_depth_metric[:, :int(A.size(3) / 3) + 5] += pred_depth_metric_l  # pred_depth_l.cpu().numpy().squeeze()  #
    pred_depth_metric[:, int(A.size(3) / 3) - 5:int(A.size(3) / 3) * 2 + 5] += pred_depth_metric_m  # pred_depth_m.cpu().numpy().squeeze()  #
    pred_depth_metric[:, int(A.size(3) / 3) * 2 - 5:] += pred_depth_metric_r  # pred_depth_r.cpu().numpy().squeeze()  #
    pred_depth_metric[:, int(A.size(3) / 3) * 2 - 5:int(A.size(3) / 3) * 2 + 5] /= 2.0
    pred_depth_metric[:, int(A.size(3) / 3) - 5:int(A.size(3) / 3) + 5] /= 2.0
    return pred_depth_metric

def test_kitti_split2(A, depth, model):
    A_l = A[:, :, :, :int(A.size(3) / 3) + 5]
    A_m = A[:, :, :, int(A.size(3) / 3) - 5:int(A.size(3) / 3) * 2 + 5]
    A_r = A[:, :, :, int(A.size(3) / 3) * 2 - 5:]

    pad_h_l = A_l.size(3) - A_l.size(2)
    pad_h_m = A_m.size(3) - A_m.size(2)
    pad_h_r = A_r.size(3) - A_r.size(2)
    A_l_pad = torch.nn.functional.pad(A_l, (0, 0, pad_h_l, 0), "constant", -5)
    A_m_pad = torch.nn.functional.pad(A_m, (0, 0, pad_h_m, 0), "constant", -5)
    A_r_pad = torch.nn.functional.pad(A_r, (0, 0, pad_h_r, 0), "constant", -5)

    pred_depth_l, _ = model.module.depth_model(A_l_pad)
    pred_depth_m, _ = model.module.depth_model(A_m_pad)
    pred_depth_r, _ = model.module.depth_model(A_r_pad)
    pred_depth_l = pred_depth_l[:, :, pad_h_l:, :]
    pred_depth_m = pred_depth_m[:, :, pad_h_m:, :]
    pred_depth_r = pred_depth_r[:, :, pad_h_r:, :]

    pred_depth_metric_l = recover_metric_depth(pred_depth_l, depth[:, :, :int(A.size(3) / 3) + 5])
    pred_depth_metric_m = recover_metric_depth(pred_depth_m, depth[:, :, int(A.size(3) / 3) - 5:int(A.size(3) / 3) * 2 + 5])
    pred_depth_metric_r = recover_metric_depth(pred_depth_r, depth[:, :, int(A.size(3) / 3) * 2 - 5:])
    pred_depth_metric = np.zeros((A.size(2), A.size(3)))
    pred_depth_metric[:, :int(A.size(3) / 3) + 5] += pred_depth_metric_l  # pred_depth_l.cpu().numpy().squeeze()  #
    pred_depth_metric[:, int(A.size(3) / 3) - 5:int(A.size(3) / 3) * 2 + 5] += pred_depth_metric_m  # pred_depth_m.cpu().numpy().squeeze()  #
    pred_depth_metric[:, int(A.size(3) / 3) * 2 - 5:] += pred_depth_metric_r  # pred_depth_r.cpu().numpy().squeeze()  #
    pred_depth_metric[:, int(A.size(3) / 3) * 2 - 5:int(A.size(3) / 3) * 2 + 5] /= 2.0
    pred_depth_metric[:, int(A.size(3) / 3) - 5:int(A.size(3) / 3) + 5] /= 2.0
    return pred_depth_metric


def test_kitti_split1(data, model):
    out = model.module.inference(data)
    pred_depth = torch.squeeze(out['b_fake'])
    pred_depth_metric = recover_metric_depth(pred_depth, data['B_raw'])
    # pred_depth = pred_depth[pred_depth.shape[0] - data['A_raw'].shape[1]:,
    #                   pred_depth.shape[1] - data['A_raw'].shape[2]:]
    return pred_depth_metric #pred_depth.cpu().numpy().squeeze()


if __name__ == '__main__':
    test_args = TestOptions().parse()
    test_args.thread = 1
    test_args.batchsize = 1
    merge_cfg_from_file(test_args)

    data_loader = CustomerDataLoader(test_args)
    test_datasize = len(data_loader)
    logger.info('{:>15}: {:<30}'.format('test_data_size', test_datasize))
    # load model
    model = RelDepthModel()

    model.eval()

    # load checkpoint
    if test_args.load_ckpt:
        load_ckpt(test_args, model)
    model.cuda()
    model = torch.nn.DataParallel(model)

    # test
    smoothed_absRel = SmoothedValue(test_datasize)
    smoothed_rms = SmoothedValue(test_datasize)
    smoothed_logRms = SmoothedValue(test_datasize)
    smoothed_squaRel = SmoothedValue(test_datasize)
    smoothed_silog = SmoothedValue(test_datasize)
    smoothed_silog2 = SmoothedValue(test_datasize)
    smoothed_log10 = SmoothedValue(test_datasize)
    smoothed_delta1 = SmoothedValue(test_datasize)
    smoothed_delta2 = SmoothedValue(test_datasize)
    smoothed_delta3 = SmoothedValue(test_datasize)
    smoothed_whdr = SmoothedValue(test_datasize)

    smoothed_criteria = {'err_absRel': smoothed_absRel, 'err_squaRel': smoothed_squaRel, 'err_rms': smoothed_rms,
                         'err_silog': smoothed_silog, 'err_logRms': smoothed_logRms, 'err_silog2': smoothed_silog2,
                         'err_delta1': smoothed_delta1, 'err_delta2': smoothed_delta2, 'err_delta3': smoothed_delta3,
                         'err_log10': smoothed_log10, 'err_whdr': smoothed_whdr}

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            A = data['A'].cuda()
            pred_depth_metric = test_kitti_split2(A, data['B_raw'], model)
            #pred_depth_metric = test_kitti_split1(data, model)

        img_path = data['A_paths']

        smoothed_criteria = evaluate_rel_err(pred_depth_metric, data['B_raw'], smoothed_criteria,  scale=80.)

        model_name = test_args.load_ckpt.split('/')[-1].split('.')[0]
        image_dir = os.path.join(cfg.ROOT_DIR, './evaluation', cfg.MODEL.ENCODER, model_name + '_KITTI')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        img_name = img_path[0].split('/')[-1]#img_path[0].split('/')[-4] + '_' +
        print(img_name)

        ################# single image error#######################
        gt = (np.squeeze(data['B_raw'].cpu().numpy())*255. * 80.)
        pred = (pred_depth_metric*255.* 80.)
        mask2 = (gt > 1e-9) & (pred_depth_metric > 1e-9)
        gt = gt[mask2]
        pred = pred[mask2]
        print(np.mean(np.abs(pred - gt)/ gt))


        #plt.imsave(os.path.join(image_dir, img_name), pred_depth_metric, cmap='rainbow')
        #cv2.imwrite(os.path.join(image_dir, img_name.replace('.', '-rgb.')), np.squeeze(data['A_raw'].cpu().numpy()))
        #plt.imsave(os.path.join(image_dir, img_name.replace('.', '-gt.')), np.squeeze(data['B_raw'].cpu().numpy()), cmap='rainbow')
        #cv2.imwrite(os.path.join(image_dir, img_name.replace('.', '.')), (pred_depth_metric*255.* 20.).astype(np.uint16))
        #cv2.imwrite(os.path.join(image_dir, img_name.replace('.', '-gtraw.')), (np.squeeze(data['B_raw'].cpu().numpy())*255. * 80.).astype(np.uint16))

        print('processing (%04d)-th image... %s' % (i, img_path))
        print("###############absREL ERROR: %f", smoothed_criteria['err_absRel'].GetGlobalAverageValue())


    print("###############WHDR ERROR: %f", smoothed_criteria['err_whdr'].GetGlobalAverageValue())
    print("###############absREL ERROR: %f", smoothed_criteria['err_absRel'].GetGlobalAverageValue())
    print("###############silog ERROR: %f", np.sqrt(smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (
        smoothed_criteria['err_silog'].GetGlobalAverageValue()) ** 2))
    print("###############log10 ERROR: %f", smoothed_criteria['err_log10'].GetGlobalAverageValue())
    print("###############RMS ERROR: %f", np.sqrt(smoothed_criteria['err_rms'].GetGlobalAverageValue()))
    print("###############delta_1 ERROR: %f", smoothed_criteria['err_delta1'].GetGlobalAverageValue())
    print("###############delta_2 ERROR: %f", smoothed_criteria['err_delta2'].GetGlobalAverageValue())
    print("###############delta_3 ERROR: %f", smoothed_criteria['err_delta3'].GetGlobalAverageValue())
    print("###############squaRel ERROR: %f", smoothed_criteria['err_squaRel'].GetGlobalAverageValue())
    print("###############logRms ERROR: %f", np.sqrt(smoothed_criteria['err_logRms'].GetGlobalAverageValue()))