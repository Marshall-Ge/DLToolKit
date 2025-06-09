
import numpy as np
import torch
import pprint
import math
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats


import dltoolkit.utils.losses as losses
import dltoolkit.utils.metrics as metrics
from dltoolkit.models.build import build_model
import dltoolkit.utils.distributed as du
import dltoolkit.utils.logging as logging
from dltoolkit.utils import misc
import dltoolkit.models.optimizer as optim
import dltoolkit.utils.checkpoint as cu
import dltoolkit.datasets.loader as loader
from dltoolkit.utils.meters import TrainMeter, ValMeter, EpochTimer

logger = logging.get_logger(__name__)

def train(cfg):

    # Set up environment.
    du.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.enabled = True

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # TODO: Init multigrid.

    # TODO: add tensorboard writer.
    writer = None
    
    # Print config.
    logger.info("--Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the model and print model statistics.
    model = build_model(cfg)
    trained_parameters = []
    for k, v in model.named_parameters():
        if v.requires_grad == True:
            trained_parameters.append(k)
    logger.info('total trainable parameters:')
    logger.info(pprint.pformat(trained_parameters))

    # TODO: caculate FLOPs and activation count.
    flops = 0.0
    params = 0.0
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        params = misc.log_model_info(model, cfg, use_train_input=True)
    
    # Construct the optimizer.
    optimizer = optim.construct_optimizer(cfg, model)

    # Load a checkpoint to resume training if applicable. use !!!
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task=cfg.TASK)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
            )
            start_epoch = checkpoint_epoch + 1
        elif "ssl_eval" in cfg.TASK:
            last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task="ssl")
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler=None,
                epoch_reset=True,
                clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            )
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = 0
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler= None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            optimizer,
            train_meter,
            cur_epoch,
            cfg,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = (
            cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
            )
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )

        is_eval_epoch = (
            misc.is_eval_epoch(
                cfg,
                cur_epoch,
            )
        )

        # Compute precise BN stats.
        if len(get_bn_modules(model)) and cfg.BN.USE_PRECISE_STATS:
            calculate_and_update_precise_bn(
                    precise_bn_loader,
                    model,
                    min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                    cfg.NUM_GPUS > 0,
                )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                train_loader,
            )
        
    if start_epoch == cfg.SOLVER.MAX_EPOCH: # eval if we loaded the final checkpoint
        eval_epoch(val_loader, model, val_meter, start_epoch, cfg, train_loader, writer)
    if writer is not None:
        writer.close()
    result_string = (
        "_p{:.2f}_f{:.2f} _t{:.2f}_m{:.2f} _a{:.2f} Top5 Acc: {:.2f} MEM: {:.2f} f: {:.4f}"
        "".format(
            params / 1e6,
            flops,
            epoch_timer.median_epoch_time() / 60.0
            if len(epoch_timer.epoch_times)
            else 0.0,
            misc.gpu_mem_usage(),
            100 - val_meter.min_top1_err,
            100 - val_meter.min_top5_err,
            misc.gpu_mem_usage(),
            flops,
        )
    )
    logger.info("training done: {}".format(result_string))

    return result_string


def train_epoch(
    train_loader,
    model,
    optimizer,
    train_meter,
    cur_epoch,
    cfg,
    writer = None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MODEL.FROZEN_BN:
        misc.frozen_bn_stats(model)
    
    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    if cfg.LOG_MODEL_INFO:
        logger.info('total trainable params:')
        for k, v in model.named_parameters():
            if v.requires_grad is True:
                logger.info('{}'.format(k))
    
    for cur_iter, (inputs, labels) in enumerate(
        train_loader
    ):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if not isinstance(labels, list):
                labels = labels.cuda(non_blocking=True)
                index = index.cuda(non_blocking=True)
        
        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        # for sure the current model contains no grad
        optimizer.zero_grad()

        preds = model(inputs)
        if isinstance(preds, list):
            preds = preds[0]
        # Compute the loss.
        loss = loss_fun(preds, labels)
        # check Nan Loss.
        misc.check_nan_losses(loss)
        
        grad_norm = optim.get_grad_norm_(model.parameters())

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, grad_norm, top1_err, top5_err = du.all_reduce(
                [loss.detach(), grad_norm, top1_err, top5_err]
            )

        # Copy the stats from GPU to CPU (sync point).
        loss, grad_norm, top1_err, top5_err = (
            loss.item(),
            grad_norm.item(),
            top1_err.item(),
            top5_err.item(),
        )

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            grad_norm,
            batch_size
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )


        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/loss": loss,
                    "Train/lr": lr,
                    "Train/Top1_err": top1_err,
                    "Train/Top5_err": top5_err,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.iter_toc()  # do measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()
    del inputs

    # in case of fragmented memory
    torch.cuda.empty_cache()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)




@torch.no_grad()
def eval_epoch(
    val_loader, model, val_meter, cur_epoch, cfg, train_loader, writer=None
):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        val_meter.data_toc()

        preds = model(inputs)
        if isinstance(preds, list):
            preds = preds[0]

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.all_reduce([top1_err, top5_err])

        # Copy the errors from GPU to CPU (sync point).
        top1_err, top5_err = top1_err.item(), top5_err.item()

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            top1_err,
            top5_err,
            batch_size
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )

        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                global_step=len(val_loader) * cur_epoch + cur_iter,
            )

        val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
        all_labels = [
            label.clone().detach() for label in val_meter.all_labels
        ]
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(
            preds=all_preds, labels=all_labels, global_step=cur_epoch
        )
    val_meter.reset()