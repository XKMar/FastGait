import os
import argparse
import sys
import time
from datetime import timedelta

import torch

sys.path.append(os.getcwd())

from fastgait.data import build_test_dataloader
from fastgait.models import build_model
from fastgait.apis.runner import test_gait
from fastgait.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from fastgait.utils.logger import Logger
from fastgait.utils.dist_utils import init_dist, synchronize
from fastgait.utils.torch_utils import copy_state_dict, load_checkpoint


def parge_config():
    parser = argparse.ArgumentParser(description="Testing Gait models")
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument("--epoch", type=str, default="40")
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()

    cfg.work_dir = args.work_dir
    args.config = os.path.join(cfg.work_dir, "config.yaml")
    cfg_from_yaml_file(args.config, cfg)
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    cfg.MODEL.sync_bn = False  # not required for inference
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    start_time = time.monotonic()

    # init distributed training
    args, cfg = parge_config()
    dist = init_dist(cfg)
    synchronize()

    # init logging file
    logger = Logger(os.path.join(cfg.work_dir, "log_test.log"))
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build model
    model = build_model(cfg)
    model.cuda()

    if dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.gpu],
            output_device=cfg.gpu,
            find_unused_parameters=True,
        )
    elif cfg.total_gpus > 1:
        model = torch.nn.DataParallel(model)

    # load checkpoint
    state_dict = load_checkpoint(os.path.join(cfg.work_dir, "checkpoint_{}.pth".format(args.epoch)))

    # load test data_loader
    test_loaders, test_datas = build_test_dataloader(cfg)

    for key in state_dict:
        if not key.startswith("state_dict"):
            continue

        print("==> Test with {}".format(key))
        copy_state_dict(state_dict[key], model)

        # start testing
        for i, (loader, test_data) in enumerate(zip(test_loaders, test_datas)):
            acc = test_gait(
                cfg, model, loader, test_data, dataset_name=cfg.TEST.datasets[i], epoch=args.epoch
            )
    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


if __name__ == "__main__":
    main()
