import importlib
import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import dill
from collections import OrderedDict
import logging
import cv2
logger = logging.getLogger(__name__)

def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'lib.models.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to f1ind function: %s', func_name)
        raise

def load_ckpt(args, model, optimizer=None, scheduler=None, val_err=[]):
    """
    Load checkpoint.
    """
    if os.path.isfile(args.load_ckpt):
        logger.info("loading checkpoint %s", args.load_ckpt)
        checkpoint = torch.load(args.load_ckpt, map_location=lambda storage, loc: storage, pickle_module=dill)
        model_state_dict_keys = model.state_dict().keys()
        checkpoint_state_dict_noprefix = strip_prefix_if_present(checkpoint['model_state_dict'], "module.")

        if all(key.startswith('module.') for key in model_state_dict_keys):
            model.module.load_state_dict(checkpoint_state_dict_noprefix)
        else:
            model.load_state_dict(checkpoint_state_dict_noprefix)
        del checkpoint
        torch.cuda.empty_cache()


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

