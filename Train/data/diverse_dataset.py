import os
import json
import torch
import torchvision.transforms as transforms
import os.path
import numpy as np
from lib.configs.config import cfg
import cv2
from lib.utils.logging import setup_logging
from torch.utils.data import Dataset
logger = setup_logging(__name__)


class DIVERSEDataset(Dataset):
    def __init__(self, opt, dataset_name=None):
        super(DIVERSEDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.dataset_name = dataset_name
        self.dir_anno = os.path.join(cfg.ROOT_DIR, opt.dataroot, dataset_name, 'annotations', opt.phase_anno + '_annotations.json')
        self.dir_teacher_list = os.path.join(cfg.ROOT_DIR, opt.dataroot, dataset_name, 'annotations', 'teacher_curriculum1.npy')
        self.rgb_paths, self.depth_paths, self.all_annos, self.curriculum_list, self.sem_masks = self.getData()
        self.data_size = len(self.all_annos)
        self.depth_normalize = 40000

    def getData(self):
        with open(self.dir_anno, 'r') as load_f:
            all_annos = json.load(load_f)
        if os.path.exists(self.dir_teacher_list):
            if 'train' in self.opt.phase:
                curriculum_list = list(np.load(self.dir_teacher_list))
                logger.info('Teacher list of %s dataset has been loaded!' %self.dataset_name)
            else:
                curriculum_list = list(np.arange(len(all_annos)))
        else:
            curriculum_list = list(np.random.choice(len(all_annos), len(all_annos), replace=False))
            logger.info('Teacher list of %s dataset does not exist!!!' % self.dataset_name)

        rgb_paths = [os.path.join(cfg.ROOT_DIR, self.root, all_annos[i]['rgb_path']) for i in range(len(all_annos))]
        depth_paths = [os.path.join(cfg.ROOT_DIR, self.root, all_annos[i]['depth_path']) for i in range(len(all_annos))]
        mask_paths = [(os.path.join(cfg.ROOT_DIR, self.root, all_annos[i]['mask_path']) if all_annos[i]['mask_path'] is not None else None)
                      for i in range(len(all_annos)) if 'mask_path' in all_annos[i]]
        return rgb_paths, depth_paths, all_annos, curriculum_list, mask_paths

    def __getitem__(self, anno_index):
        if 'train' in self.opt.phase:
            data = self.online_aug(anno_index)
        else:
            data = self.load_test_data(anno_index)
        return data

    def load_test_data(self, anno_index):
        """
        Augment data for training online randomly. The invalid parts in the depth map are set to -1.0, while the parts
        in depth bins are set to cfg.MODEL.DECODER_OUTPUT_C + 1.
        :param anno_index: data index.
        """
        rgb_path = self.rgb_paths[anno_index]
        depth_path = self.depth_paths[anno_index]

        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # bgr, H*W*C
        depth, sky_mask, mask_valid = self.load_depth(anno_index, rgb)

        rgb_resize = cv2.resize(rgb, (cfg.DATASET.CROP_SIZE[1], cfg.DATASET.CROP_SIZE[0]),
                              interpolation=cv2.INTER_LINEAR)
        # to torch, normalize
        A_torch = self.scale_torch(rgb_resize.copy())
        # normalize disp and depth
        depth_normal = depth / (depth.max() + 1e-8)
        depth_normal[~mask_valid.astype(np.bool)] = 0

        data = {'A': A_torch, 'A_raw': rgb.copy(), 'B_raw': depth_normal, 'A_path': rgb_path,
                'B_path': depth_path}
        return data

    def online_aug(self, anno_index):
        """
        Augment data for training online randomly.
        :param anno_index: data index.
        """
        rgb_path = self.rgb_paths[anno_index]
        depth_path = self.depth_paths[anno_index]
        rgb = cv2.imread(rgb_path)[:, :, ::-1]   # rgb, H*W*C
        depth, sky_mask, mask_valid = self.load_depth(anno_index, rgb)

        flip_flg, resize_size, crop_size, pad, resize_ratio = self.set_flip_resize_crop_pad(rgb)

        rgb_resize = self.flip_reshape_crop_pad(rgb, flip_flg, resize_size, crop_size, pad, 0)
        depth_resize = self.flip_reshape_crop_pad(depth, flip_flg, resize_size, crop_size, pad, -1)

        # mask all invalid parts of gt depth
        mask_valid_resize = self.flip_reshape_crop_pad(mask_valid, flip_flg, resize_size, crop_size, pad, 0)
        mask_valid_resize[mask_valid_resize<1.0] = 0

        sky_mask_resize = self.flip_reshape_crop_pad(sky_mask.astype(np.uint8), flip_flg, resize_size, crop_size, pad, 0)

        # normalize disp and depth
        depth_resize = depth_resize / (depth_resize.max() + 1e-8) * 10
        # invalid regions are set to -1, sky regions are set to 0 in disp and 10 in depth
        depth_resize[~mask_valid_resize.astype(np.bool) | (depth_resize > 1e7) | (depth_resize < 0)] = -1
        depth_resize[sky_mask_resize.astype(np.bool)] = 100

        # to torch, normalize
        rgb_torch = self.scale_torch(rgb_resize.copy())
        depth_torch = self.scale_torch(depth_resize)

        data = {'A': rgb_torch, 'B': depth_torch, 'A_paths': rgb_path, 'B_paths': depth_path}
        return data

    def set_flip_resize_crop_pad(self, A):
        """
        Set flip, padding, reshaping and cropping flags.
        :param A: Input image, [H, W, C]
        :return: Data augamentation parameters
        """
        # flip
        flip_prob = np.random.uniform(0.0, 1.0)
        flip_flg = True if flip_prob > 0.5 and 'train' in self.opt.phase else False

        # reshape
        ratio_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  #
        if 'train' in self.opt.phase:
            resize_ratio = ratio_list[np.random.randint(len(ratio_list))]
        else:
            resize_ratio = 0.5

        resize_size = [int(A.shape[0] * resize_ratio + 0.5),
                       int(A.shape[1] * resize_ratio + 0.5)]  # [height, width]
        # crop
        start_y = 0 if resize_size[0] <= cfg.DATASET.CROP_SIZE[0] else np.random.randint(0, resize_size[0] - cfg.DATASET.CROP_SIZE[0])
        start_x = 0 if resize_size[1] <= cfg.DATASET.CROP_SIZE[1] else np.random.randint(0, resize_size[1] - cfg.DATASET.CROP_SIZE[1])
        crop_height = resize_size[0] if resize_size[0] <= cfg.DATASET.CROP_SIZE[0] else cfg.DATASET.CROP_SIZE[0]
        crop_width = resize_size[1] if resize_size[1] <= cfg.DATASET.CROP_SIZE[1] else cfg.DATASET.CROP_SIZE[1]
        crop_size = [start_x, start_y, crop_width, crop_height] if 'train' in self.opt.phase else [0, 0, resize_size[1], resize_size[0]]

        # pad
        pad_height = 0 if resize_size[0] > cfg.DATASET.CROP_SIZE[0] else cfg.DATASET.CROP_SIZE[0] - resize_size[0]
        pad_width = 0 if resize_size[1] > cfg.DATASET.CROP_SIZE[1] else cfg.DATASET.CROP_SIZE[1] - resize_size[1]
        # [up, down, left, right]
        pad = [pad_height, 0, pad_width, 0] if 'train' in self.opt.phase else [0, 0, 0, 0]
        return flip_flg, resize_size, crop_size, pad, resize_ratio

    def flip_reshape_crop_pad(self, img, flip, resize_size, crop_size, pad, pad_value=0):
        """
        Flip, pad, reshape, and crop the image.
        :param img: input image, [C, H, W]
        :param flip: flip flag
        :param crop_size: crop size for the image, [x, y, width, height]
        :param pad: pad the image, [up, down, left, right]
        :param pad_value: padding value
        :return:
        """
        # Flip
        if flip:
            img = np.flip(img, axis=1)

        # Resize the raw image
        img_resize = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)

        # Crop the resized image
        img_crop = img_resize[crop_size[1]:crop_size[1] + crop_size[3], crop_size[0]:crop_size[0] + crop_size[2]]

        # Pad the raw image
        if len(img.shape) == 3:
            img_pad = np.pad(img_crop, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
                             constant_values=(pad_value, pad_value))
        else:
            img_pad = np.pad(img_crop, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
                             constant_values=(pad_value, pad_value))
        return img_pad

    def depth_to_bins(self, depth):
        """
        Discretize depth into depth bins
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: depth bins [1, h, w]
        """
        invalid_mask = depth < 1e-8
        depth[depth < cfg.DATASET.DEPTH_MIN] = cfg.DATASET.DEPTH_MIN
        depth[depth > cfg.DATASET.DEPTH_MAX] = cfg.DATASET.DEPTH_MAX
        bins = ((torch.log10(depth) - cfg.DATASET.DEPTH_MIN_LOG) / cfg.DATASET.DEPTH_BIN_INTERVAL).to(torch.int)
        bins[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
        bins[bins == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
        depth[invalid_mask] = -1.0
        return bins

    def scale_torch(self, img):
        """
        Scale the image and output it in torch.tensor.
        :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W]
        """
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        if img.shape[2] == 3:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)])
            img = transform(img)
        else:
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
        return img

    def load_depth(self, anno_index, rgb):
        """
        Load disparity, depth, and mask maps
        :return
            disp: disparity map,  np.float
            depth: depth map, np.float
            sem_mask: semantic masks, including sky, road, np.uint8
            ins_mask: plane instance masks, np.uint8
        """
        # load depth
        depth = cv2.imread(self.depth_paths[anno_index], -1)
        depth, mask_valid = self.preprocess_depth(depth, self.depth_paths[anno_index])

        # load semantic mask, such as road, sky
        if len(self.rgb_paths) == len(self.sem_masks) and self.sem_masks[anno_index] is not None:
            sem_mask = cv2.imread(self.sem_masks[anno_index], -1).astype(np.uint8)
        else:
            sem_mask = np.zeros(depth.shape, dtype=np.uint8)
        sky_mask = sem_mask == 17

        return depth, sky_mask, mask_valid

    def preprocess_depth(self, depth, img_path):
        if 'diml' in img_path.lower():
            drange = 65535.0
        elif 'taskonomy' in img_path.lower():
            depth[depth > 23000] = 0
            drange = 23000.0
        elif 'diversedepth' in img_path.lower():
            #depth_filter1 = depth[depth > 1e-8]
            #drange = (depth_filter1.max() - depth_filter1.min())
            drange = depth.max()
        else:
            raise RuntimeError('Unknown dataset!')
        depth_norm = depth / drange
        mask_valid = (depth_norm > 1e-8).astype(np.float)
        return depth_norm, mask_valid

    def __len__(self):
        return self.data_size

    def name(self):
        return 'DiverseDepth'

