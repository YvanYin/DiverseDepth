import torch.utils.data
import importlib
import numpy as np
# from lib.utils.logging import setup_logging
# logger = setup_logging(__name__)

import logging
logger = logging.getLogger(__name__)

class CustomerDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = create_dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchsize,
            shuffle=True if 'train' in opt.phase else False,
            num_workers=opt.thread)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchsize >= float("inf"):
                break
            yield data


class MultipleDataLoader():
    def __init__(self, opt, sample_ratio=0.2):
        self.opt = opt
        self.multi_datasets, self.dataset_indices_list = create_multiple_dataset(opt)
        self.datasizes = [len(dataset) for dataset in self.multi_datasets]
        self.merged_dataset = torch.utils.data.ConcatDataset(self.multi_datasets)
        self.custom_multi_sampler = CustomerSamples(self.dataset_indices_list, sample_ratio, self.opt)
        self.curr_sample_size = self.custom_multi_sampler.num_samples
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.merged_dataset,
            batch_size=opt.batchsize,
            num_workers=opt.thread,
            sampler=self.custom_multi_sampler)

    def load_data(self):
        return self

    def __len__(self):
        return np.sum(self.datasizes)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchsize >= float("inf"):
                break
            yield data


class CustomerSamples(torch.utils.data.Sampler):
    """
    Construct a sample method. Sample former ratio_samples of datasets randomly.
    """
    def __init__(self, multi_data_indices, ratio_samples, opt):
        self.multi_data_indices = multi_data_indices
        self.num_indices = np.array([len(i) for i in self.multi_data_indices])
        self.num_samples = (self.num_indices * ratio_samples).astype(np.uint32)
        self.max_indices = np.array([max(i) for i in self.multi_data_indices])
        self.phase = opt.phase
        logger.info('Sample %02f, sampled dataset sizes are %s' % (ratio_samples, ','.join(map(str, self.num_samples))))

    def __iter__(self):
        cum_sum = np.cumsum(np.append([0], self.max_indices))
        if 'train' in self.phase:
            indices_array = [[self.multi_data_indices[idx][i] + cum_sum[idx] for i in torch.randperm(int(num))] for idx, num in
                          enumerate(self.num_samples)]
        else:
            indices_array = [[self.multi_data_indices[idx][i] + cum_sum[idx] for i in range(int(num))] for
                             idx, num in enumerate(self.num_samples)]
        if 'train' in self.phase:
            # data list is reshaped in [A, B, C, A, B, C....]
            indices_array = np.array(indices_array).transpose(1, 0).reshape(-1)
        else:
            indices_array = np.concatenate(indices_array[:])
        return iter(indices_array)


def create_dataset(opt):
    dataset = find_dataset_lib(opt.dataset)()
    dataset.initialize(opt)
    logger.info("%s is created." % opt.dataset)
    return dataset


def create_multiple_dataset(opt):
    all_datasets = []
    dataset_indices_lists = []
    indices_len = []
    for name in opt.dataset_list:
        dataset = find_dataset_lib(opt.dataset)()
        dataset.initialize(opt, name)
        logger.info("%s : %s is loaded, the data size is %d" % (opt.phase, name, len(dataset)))
        all_datasets.append(dataset)
        assert dataset.teacher_list is not None, "Curriculum is None!!!"
        dataset_indices_lists.append(dataset.teacher_list)
        indices_len.append(len(dataset.teacher_list))
        assert len(dataset.teacher_list) == dataset.data_size, "Curriculum list size not equal the data size!!!"
    max_len = np.max(indices_len)
    if 'train' in opt.phase:
        extended_indices_list = [i + list(np.random.choice(i, max_len-len(i))) for i in dataset_indices_lists]
    else:
        extended_indices_list = dataset_indices_lists
    logger.info("%s are merged!" % opt.dataset_list)
    return all_datasets, extended_indices_list


def find_dataset_lib(dataset_name):
    """
    Give the option --dataset [datasetname], import "data/datasetname_dataset.py"
    :param dataset_name: --dataset
    :return: "data/datasetname_dataset.py"
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls
    if dataset is None:
        logger.info("In %s.py, there should be a class name that matches %s in lowercase." % (
        dataset_filename, target_dataset_name))
        exit(0)
    return dataset