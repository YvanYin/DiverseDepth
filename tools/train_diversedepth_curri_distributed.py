from data.load_dataset_distributed import MultipleDataLoaderDistributed
from lib.utils.training_stats import TrainingStats
from lib.utils.evaluate_depth_error import validate_rel_depth_err, recover_metric_depth
from lib.models.fill_depth_model_distributed import *
from lib.core.config import cfg, merge_cfg_from_file, print_configs
from lib.utils.net_tools import save_ckpt, load_ckpt
from lib.utils.logging import setup_distributed_logger, SmoothedValue
import math
import traceback
from tools.parse_arg_train import TrainOptions
from tools.parse_arg_val import ValOptions
import torch.distributed
from lib.utils.comm import get_world_size, synchronize, get_rank, is_pytorch_1_1_0_or_later
import errno


import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cfg.TRAIN.LOG_DIR)
if BASE_DIR:
    try:
        os.makedirs(BASE_DIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
sys.stdout = open(os.path.join(BASE_DIR, cfg.TRAIN.RUN_NAME + '.txt'), 'w')


def increase_sample_ratio_steps(step, base_ratio=0.1, step_size=10000):
    ratio = min(base_ratio * (int(step / step_size) + 1), 1.0)
    return ratio

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        torch.distributed.reduce(all_losses, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def val(val_dataloader, model):
    """
    Validate the model.
    """
    smoothed_absRel = SmoothedValue(len(val_dataloader))
    smoothed_whdr = SmoothedValue(len(val_dataloader))
    smoothed_criteria = {'err_absRel': smoothed_absRel, 'err_whdr': smoothed_whdr}
    for i, data in enumerate(val_dataloader):
        invalid_side = data['invalid_side'][0]
        out = model.module.inference(data)
        pred_depth = torch.squeeze(out['b_fake'])

        pred_depth = pred_depth[invalid_side[0]:pred_depth.size(0) - invalid_side[1],
                                invalid_side[2]:pred_depth.size(1) - invalid_side[3]]

        pred_depth_resize = resize_image(pred_depth, torch.squeeze(data['B_raw']).shape)
        pred_depth_metric = recover_metric_depth(pred_depth_resize, data['B_raw'])
        smoothed_criteria = validate_rel_depth_err(pred_depth_metric, data['B_raw'], smoothed_criteria, scale=1.0)
    return {'abs_rel': smoothed_criteria['err_absRel'].GetGlobalAverageValue(),
            'whdr': smoothed_criteria['err_whdr'].GetGlobalAverageValue()}


def train(local_rank, distributed, train_args, logger, tblogger=None):
    # load model
    model = RelDepthModel()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Optimizer
    optimizer = ModelOptimizer(model)
    #lr_optim_lambda = lambda iter: (1.0 - iter / (float(total_iters))) ** 0.9
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, lr_lambda=lr_optim_lambda)
    lr_scheduler_step = 15000
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizer, step_size=lr_scheduler_step, gamma=0.9)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)

    val_err = [{'abs_rel': 0, 'whdr': 0}]

    # training and validation dataloader
    val_dataloader = MultipleDataLoaderDistributed(val_args)
    if train_args.load_ckpt:
        load_ckpt(train_args, model, optimizer.optimizer, scheduler, val_err)
        # obtain the current sample ratio
        sample_ratio = increase_sample_ratio_steps(train_args.start_step, base_ratio=train_args.sample_start_ratio,
                                                   step_size=train_args.sample_ratio_steps)
        # reconstruct the train_dataloader with the new sample_ratio
        train_dataloader = MultipleDataLoaderDistributed(train_args, sample_ratio=sample_ratio)
        if not train_args.resume:
            scheduler.__setattr__('step_size', lr_scheduler_step)
    else:
        train_dataloader = MultipleDataLoaderDistributed(train_args)

    train_datasize = len(train_dataloader)
    val_datasize = len(val_dataloader)
    merge_cfg_from_file(train_args)

    # total iterations
    total_iters = math.ceil(train_datasize / train_args.batchsize) * train_args.epoch
    cfg.TRAIN.MAX_ITER = total_iters
    cfg.TRAIN.GPU_NUM = gpu_num

    # Print configs and logs
    if get_rank() == 0:
        train_opt.print_options(train_args)
        val_opt.print_options(val_args)
        print_configs(cfg)
        logger.info('{:>15}: {:<30}'.format('GPU_num', gpu_num))
        logger.info('{:>15}: {:<30}'.format('train_data_size', train_datasize))
        logger.info('{:>15}: {:<30}'.format('val_data_size', val_datasize))
        logger.info('{:>15}: {:<30}'.format('total_iterations', total_iters))

    save_to_disk = get_rank() == 0

    do_train(train_dataloader,
             val_dataloader,
             train_args,
             model,
             save_to_disk,
             scheduler,
             optimizer,
             val_err,
             logger,
             tblogger)

def do_train(train_dataloader, val_dataloader, train_args,
             model, save_to_disk,
             scheduler, optimizer, val_err,
             logger, tblogger=None):

    # training status for logging
    if save_to_disk:
        training_stats = TrainingStats(train_args, cfg.TRAIN.LOG_INTERVAL,
                                   tblogger if train_args.use_tfboard else None)

    dataloader_iterator = iter(train_dataloader)
    start_step = train_args.start_step
    total_iters = cfg.TRAIN.MAX_ITER
    train_datasize = len(train_dataloader)

    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()

    try:
        for step in range(start_step, total_iters):

            if step % train_args.sample_ratio_steps == 0 and step != 0:
                sample_ratio = increase_sample_ratio_steps(step, base_ratio=train_args.sample_start_ratio, step_size=train_args.sample_ratio_steps)
                train_dataloader = MultipleDataLoaderDistributed(train_args, sample_ratio=sample_ratio)
                dataloader_iterator = iter(train_dataloader)
                logger.info('Sample ratio: %02f, current sampled datasize: %d' % (sample_ratio, np.sum(train_dataloader.curr_sample_size)))

            # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
            if not pytorch_1_1_0_or_later:
                scheduler.step()

            epoch = int(step * train_args.batchsize / train_datasize)
            if save_to_disk:
                training_stats.IterTic()

            # get the next data batch
            try:
                data = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_dataloader)
                data = next(dataloader_iterator)

            out = model(data)
            losses_dict = out['losses']
            optimizer.optim(losses_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(losses_dict)

            if pytorch_1_1_0_or_later:
                scheduler.step()
            if save_to_disk:
                training_stats.UpdateIterStats(loss_dict_reduced)
                training_stats.IterToc()
                training_stats.LogIterStats(step, epoch, optimizer.optimizer, val_err[0])

            # validate the model
            if step % cfg.TRAIN.VAL_STEP == 0 and val_dataloader is not None and step != 0:
                model.eval()
                val_err[0] = val(val_dataloader, model)
                # training mode
                model.train()
            # save checkpoint
            if step % cfg.TRAIN.SNAPSHOT_ITERS == 0 and step != 0 and save_to_disk:
                save_ckpt(train_args, step, epoch, model, optimizer.optimizer, scheduler, val_err[0])

    except (RuntimeError, KeyboardInterrupt):
        stack_trace = traceback.format_exc()
        print(stack_trace)
    finally:
        if train_args.use_tfboard and get_rank()==0:
            tblogger.close()


if __name__=='__main__':
    # Train args
    train_opt = TrainOptions()
    train_args = train_opt.parse()

    # Validation args
    val_opt = ValOptions()
    val_args = val_opt.parse()
    val_args.batchsize = 1
    val_args.thread = 0

    gpu_num = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    train_args.distributed = gpu_num > 1

    # set distributed configs
    if train_args.distributed:
        torch.cuda.set_device(train_args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        synchronize()

    # Set logger
    log_output_dir = cfg.TRAIN.LOG_DIR
    if log_output_dir:
        try:
            os.makedirs(log_output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    logger = setup_distributed_logger("lib", log_output_dir, get_rank(), cfg.TRAIN.RUN_NAME + '.txt')
    # tensorboard logger
    tblogger = None
    if train_args.use_tfboard and get_rank() == 0:
        from tensorboardX import SummaryWriter

        tblogger = SummaryWriter(cfg.TRAIN.LOG_DIR)

    merge_cfg_from_file(train_args)
    train(train_args.local_rank, train_args.distributed, train_args, logger, tblogger)

