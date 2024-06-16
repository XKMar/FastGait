import os
import sys
import time
import shutil
import argparse

from datetime import timedelta

import torch

sys.path.append(os.getcwd())

from fastgait.apis.runner import BaseRunner
from fastgait.apis.train import set_random_seed
from fastgait.solvers import build_lr_scheduler, build_optimizer
from fastgait.data import build_train_dataloader
from fastgait.models import build_model
from fastgait.models.losses import build_loss
from fastgait.utils.config import (
cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,)
from fastgait.utils.dist_utils import init_dist, synchronize
from fastgait.utils.file_utils import model_path
from fastgait.utils.logger import Logger

def parge_config():
    parser = argparse.ArgumentParser(description="Base Neural Network Training for Gait")
    parser.add_argument("--config", default='config.yaml', help="train config file path")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()

    # update cfg from config.yaml
    cfg_from_yaml_file(args.config, cfg)
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    
    # work dir 
    cfg.work_dir = model_path(cfg.LOGS_ROOT, 
                        cfg.DATA.name, 
                        cfg.MODEL.meta_arch,
                        cfg.MODEL.back_bone)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    shutil.copy(args.config, os.path.join(cfg.work_dir, "config.yaml"))

    return args, cfg


def main():
    start_time = time.monotonic()

    # assign parameters to variables
    args, cfg = parge_config()
    dist = init_dist(cfg)
    set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)
    synchronize()

    # init logging file
    log_name = "{}_{}".format(cfg.MODEL.meta_arch, 
                              cfg.DATA.name) + '.log'
    logger = Logger(os.path.join(cfg.work_dir, log_name), debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build train loader
    train_loader, train_sets = build_train_dataloader(cfg)

    # build model
    model = build_model(cfg, init=cfg.MODEL.pretrained)

    # init Distributed Training
    if dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.gpu],
            output_device=cfg.gpu,
            find_unused_parameters=False,
        )
    elif cfg.total_gpus > 1:
        model = torch.nn.DataParallel(model)

    # build optimizer
    optimizer = build_optimizer([model,], **cfg.TRAIN.OPTIM)

    # build lr_scheduler
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
    else:
        lr_scheduler = None

    # build loss functions
    criterions = build_loss(
                cfg.TRAIN.LOSS, 
                num_classes=cfg.MODEL.num_classes, 
                margin=cfg.TRAIN.LOSS.margin,
                cuda=True)

    # build runner
    runner = BaseRunner(
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        train_sets=train_sets,
        lr_scheduler=lr_scheduler,
        reset_optim=True,
    )

    # resume
    if args.resume_from:
        runner.resume(args.resume_from)

    # start training
    runner.run()

    # print time
    shutil.copy(args.config, os.path.join(cfg.work_dir, "config.yaml"))
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


if __name__ == "__main__":
    main()
