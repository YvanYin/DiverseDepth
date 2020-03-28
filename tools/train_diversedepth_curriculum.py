from data.load_dataset import MultipleDataLoader
from lib.utils.training_stats import TrainingStats
from lib.utils.evaluate_depth_error import validate_rel_depth_err, recover_metric_depth
from lib.models.diverse_depth_model import *
from lib.core.config import cfg, merge_cfg_from_file, print_configs
from lib.utils.net_tools import save_ckpt, load_ckpt
from lib.utils.logging import setup_logger, SmoothedValue
import math
import traceback
from tools.parse_arg_train import TrainOptions
from tools.parse_arg_val import ValOptions
import errno


# import sys
# BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cfg.TRAIN.LOG_DIR)
# if not os.path.exists(BASE_DIR):
#     os.makedirs(BASE_DIR)
# sys.stdout = open(os.path.join(BASE_DIR, cfg.TRAIN.RUN_NAME + '.txt'), 'w')



def increase_sample_ratio_steps(step, base_ratio=0.1, step_size=10000):
    ratio = min(base_ratio * (int(step / step_size) + 1), 1.0)
    return ratio


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


if __name__=='__main__':
    # Train args
    train_opt = TrainOptions()
    train_args = train_opt.parse()
    merge_cfg_from_file(train_args)

    gpu_num = torch.cuda.device_count()
    cfg.TRAIN.GPU_NUM = gpu_num

    # Validation args
    val_opt = ValOptions()
    val_args = val_opt.parse()
    val_args.batchsize = 1
    val_args.thread = 0

    # Logger
    log_output_dir = cfg.TRAIN.LOG_DIR
    if log_output_dir:
        try:
            os.makedirs(log_output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    logger = setup_logger("lib", log_output_dir, cfg.TRAIN.RUN_NAME + '.txt')

    # tensorboard logger
    if train_args.use_tfboard:
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(cfg.TRAIN.LOG_DIR)

     # training status for logging
    training_stats = TrainingStats(train_args, cfg.TRAIN.LOG_INTERVAL,
                                   tblogger if train_args.use_tfboard else None)

    val_err = [{'abs_rel': 0, 'whdr': 0}]
    start_step = 0

    # load model, config optimizer
    model = RelDepthModel()
    model.cuda()
    optimizer = ModelOptimizer(model)
    loss_func = ModelLoss()

    # Optimizer
    lr_optim_lambda = lambda iter: (1.0 - iter / (float(total_iters))) ** 0.9
    lr_scheduler_step = 15000
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizer, step_size=lr_scheduler_step, gamma=0.9)


    # load checkpoint
    if train_args.load_ckpt:
        load_ckpt(train_args, model, optimizer.optimizer, scheduler, val_err)
        # obtain the current sample ratio
        sample_ratio = increase_sample_ratio_steps(train_args.start_step, base_ratio=train_args.sample_start_ratio,
                                                   step_size=train_args.sample_ratio_steps)
        # reconstruct the train_dataloader with the new sample_ratio
        train_dataloader = MultipleDataLoader(train_args, sample_ratio=sample_ratio)
        start_step = train_args.start_step
        if not train_args.resume:
            scheduler.__setattr__('step_size', lr_scheduler_step)
    else:
        train_dataloader = MultipleDataLoader(train_args, sample_ratio=train_args.sample_start_ratio)

    val_dataloader = MultipleDataLoader(val_args, sample_ratio=1)
    val_datasize = len(val_dataloader)
    train_datasize = len(train_dataloader)
    train_indices_size = len(train_dataloader.datasizes) * max(train_dataloader.datasizes)
    # total iterations
    total_iters = math.ceil(train_indices_size / train_args.batchsize) * train_args.epoch
    cfg.TRAIN.MAX_ITER = total_iters

    dataloader_iterator = iter(train_dataloader)

    # Print configs
    merge_cfg_from_file(train_args)
    print_configs(cfg)
    train_opt.print_options(train_args)
    val_opt.print_options(val_args)
    if gpu_num != -1:
        logger.info('----------------- Training Information ---------------')
        logger.info('{:>15}: {:<30}'.format('GPU_num', gpu_num))
        logger.info('{:>15}: {:<30}'.format('train_data_size', train_datasize))
        logger.info('{:>15}: {:<30}'.format('train_indices_size', train_indices_size))
        logger.info('{:>15}: {:<30}'.format('val_data_size', val_datasize))
        logger.info('{:>15}: {:<30}'.format('total_iterations', total_iters))
        logger.info('----------------- End ---------------')


    if gpu_num != -1:
        model = torch.nn.DataParallel(model)

    try:
        for step in range(start_step, total_iters):
            scheduler.step()  # decay lr every iteration
            epoch = int(step * train_args.batchsize / train_datasize)

            if step % train_args.sample_ratio_steps == 0 and step != 0:
                sample_ratio = increase_sample_ratio_steps(step, base_ratio=train_args.sample_start_ratio, step_size=train_args.sample_ratio_steps)
                train_dataloader = MultipleDataLoader(train_args, sample_ratio=sample_ratio)
                dataloader_iterator = iter(train_dataloader)
                logger.info('Sample ratio: %02f, current sampled datasize: %d' % (sample_ratio, np.sum(train_dataloader.curr_sample_size)))

            # get the next data batch
            try:
                data = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_dataloader)
                data = next(dataloader_iterator)

            training_stats.IterTic()
            out = model(data)
            losses = loss_func.criterion(out['b_fake_softmax'], out['b_fake_logit'], data)
            optimizer.optim(losses)
            training_stats.UpdateIterStats(losses)
            training_stats.IterToc()
            training_stats.LogIterStats(step, 0, optimizer.optimizer, val_err[0])
            # validate the model
            if (step+1) % cfg.TRAIN.VAL_STEP == 0  and val_dataloader is not None and step != 0:
                model.eval()
                val_err[0] = val(val_dataloader, model)
                # training mode
                model.train()
            # save checkpoint
            if step % cfg.TRAIN.SNAPSHOT_ITERS == 0 and step != 0:
                save_ckpt(train_args, step, epoch, model, optimizer.optimizer, scheduler, val_err[0])


    except (RuntimeError, KeyboardInterrupt):
        logger.info('Save ckpt on exception ...')
        stack_trace = traceback.format_exc()
        print(stack_trace)
    finally:
        if train_args.use_tfboard:
            tblogger.close()
