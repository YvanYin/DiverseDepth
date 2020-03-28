from tools.parse_arg_test import TestOptions
from data.load_dataset import CustomerDataLoader
from lib.models.diverse_depth_model import RelDepthModel
from lib.utils.net_tools import load_ckpt
import torch
import os
import numpy as np
from lib.core.config import cfg, merge_cfg_from_file
import matplotlib.pyplot as plt
from lib.utils.logging import setup_logging, SmoothedValue

import torchvision.transforms as transforms
from lib.utils.evaluate_depth_error import evaluate_rel_err, recover_metric_depth
logger = setup_logging(__name__)
import cv2
import json

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

    anno_path = './annotations.json' # add the path of annotations here, annotations contains the path to rgb and depth maps.
    base_path = os.path.dirname(os.path.dirname(anno_path))
    f = open(anno_path, 'r')
    annos = json.load(f)

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

    for i, v in enumerate(annos):
        print(i)
        with torch.no_grad():
            print('processing (%04d)-th image... %s' % (i, v['rgb_path'].strip()))
            img_path = os.path.join(base_path, v['rgb_path'])
            depth_path = os.path.join(base_path, v['depth_path'])
            rgb = cv2.imread(img_path)
            #depth = cv2.imread(depth_path, -1)

            img_torch = scale_torch(rgb, 255)
            img_torch = img_torch[None, :, :, :].cuda()

            pred_depth, _ = model.module.depth_model(img_torch)
            #pred_depth = torch.nn.functional.tanh(pred_depth) + 1

            pred_depth = pred_depth.cpu().numpy().squeeze()
            #pred_depth_metric = recover_metric_depth(pred_depth, depth)

           # smoothed_criteria = evaluate_rel_err(pred_depth_metric, depth, smoothed_criteria, scale=1.0)

            model_name = test_args.load_ckpt.split('/')[-1].split('.')[0]
            image_dir = os.path.join(cfg.ROOT_DIR, './evaluation', cfg.MODEL.ENCODER, model_name + '_anyimgs')
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            img_name = v['rgb_path'].split('/')[1]
            #plt.imsave(os.path.join(image_dir, img_name.replace('.', '-c.')), pred_depth, cmap='rainbow')
            #cv2.imwrite(os.path.join(image_dir, img_name), rgb)
            print('processing (%04d)-th image... %s' % (i, v['rgb_path'].strip()))

    # print("###############WHDR ERROR: %f", smoothed_criteria['err_whdr'].GetGlobalAverageValue())
    # print("###############absREL ERROR: %f", smoothed_criteria['err_absRel'].GetGlobalAverageValue())
    # print("###############silog ERROR: %f", np.sqrt(smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (
    #         smoothed_criteria['err_silog'].GetGlobalAverageValue()) ** 2))
    # print("###############log10 ERROR: %f", smoothed_criteria['err_log10'].GetGlobalAverageValue())
    # print("###############RMS ERROR: %f", np.sqrt(smoothed_criteria['err_rms'].GetGlobalAverageValue()))
    # print("###############delta_1 ERROR: %f", smoothed_criteria['err_delta1'].GetGlobalAverageValue())
    # print("###############delta_2 ERROR: %f", smoothed_criteria['err_delta2'].GetGlobalAverageValue())
    # print("###############delta_3 ERROR: %f", smoothed_criteria['err_delta3'].GetGlobalAverageValue())
    # print("###############squaRel ERROR: %f", smoothed_criteria['err_squaRel'].GetGlobalAverageValue())
    # print("###############logRms ERROR: %f", np.sqrt(smoothed_criteria['err_logRms'].GetGlobalAverageValue()))
    # print('\n')