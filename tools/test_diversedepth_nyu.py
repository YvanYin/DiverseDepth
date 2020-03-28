from tools.parse_arg_test import TestOptions
from data.load_dataset import CustomerDataLoader
from lib.models.diverse_depth_model import RelDepthModel
from lib.utils.net_tools import load_ckpt
import torch
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
        out = model.module.inference(data)
        pred_depth = torch.squeeze(out['b_fake'])

        invalid_side = data['invalid_side'][0]
        pred_depth = pred_depth[invalid_side[0]:pred_depth.size(0) - invalid_side[1], :]
        pred_depth_resize = resize_image(pred_depth, torch.squeeze(data['B_raw']).shape)

        img_path = data['A_paths']
        # Recover metric depth
        pred_depth_metric = recover_metric_depth(pred_depth_resize, data['B_raw'])

        smoothed_criteria = evaluate_rel_err(pred_depth_metric, data['B_raw'], smoothed_criteria, mask=(45, 471, 41, 601), scale=10.)

        model_name = test_args.load_ckpt.split('/')[-1].split('.')[0]
        image_dir = os.path.join(cfg.ROOT_DIR, './evaluation', cfg.MODEL.ENCODER, model_name + '_nyu')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        img_name = img_path[0].split('/')[-1]
        depth_max = pred_depth_metric.max()

        ################# single image error#######################
        gt = (np.squeeze(data['B_raw'].cpu().numpy()) * 10 * 6000.)
        pred = (pred_depth_metric * 10 * 6000.)

        mask2 = (gt > 1e-9) & (pred_depth_metric > 1e-9)
        gt = gt[mask2]
        pred = pred[mask2]
        print(img_name)
        print(np.mean((np.abs(pred - gt) / gt)))
        ###########################################################

        #plt.imsave(os.path.join(image_dir, img_name), pred_depth_metric, cmap='rainbow')
        #cv2.imwrite(os.path.join(image_dir, img_name.replace('.', '-rgb.')), np.squeeze(data['A_raw'].cpu().numpy()))
        #plt.imsave(os.path.join(image_dir, img_name.replace('.', '-gt.')), np.squeeze(data['B_raw'].cpu().numpy()), cmap='rainbow')
        #cv2.imwrite(os.path.join(image_dir, img_name.replace('.', '-raw.')), (pred_depth_metric * 60000).astype(np.uint16))
        #cv2.imwrite(os.path.join(image_dir, img_name.replace('.', '-gtraw.')), (np.squeeze(data['B_raw'].cpu().numpy() * 60000).astype(np.uint16)))

        #print('processing (%04d)-th image... %s' % (i, img_path))
        #print("###############absREL ERROR: %f", smoothed_criteria['err_absRel'].GetGlobalAverageValue())
        #print("###############silog ERROR: %f", np.sqrt(smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (
        #smoothed_criteria['err_silog'].GetGlobalAverageValue()) ** 2))

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