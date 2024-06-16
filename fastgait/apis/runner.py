import time
import os.path as osp

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel

from fastgait.utils.meters import Meters
from fastgait.utils.dist_utils import get_dist_info, synchronize
from fastgait.apis.test import test_gait
from fastgait.apis.train import batch_processor
from fastgait.data import build_test_dataloader
from fastgait.utils.torch_utils import copy_state_dict, load_checkpoint, save_checkpoint

class BaseRunner(object):
    """
        Base Runner for Gait Recognition.
        A subclass can redesign "train_step".
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        train_sets=None,
        lr_scheduler=None,
        meter_formats=None,
        print_freq=100,
        reset_optim=True,
    ):
        super(BaseRunner, self).__init__()

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterions = criterions
        self.lr_scheduler = lr_scheduler
        self.print_freq = print_freq
        self.reset_optim = reset_optim

        self._rank, self._world_size, self._is_dist = get_dist_info()
        self._epoch, self._start_epoch = 0, 0

        # build data loaders
        self.train_loader, self.train_sets = train_loader, train_sets
        if "val_dataset" in self.cfg.TRAIN:
            self.val_loader, self.val_set = build_test_dataloader(cfg)

        # build train progress for meters
        if meter_formats is None:
            meter_formats = {"Time": ":.3f"}
        self.train_progress = Meters(meter_formats, 
                    self.cfg.TRAIN.iters, prefix="Train: ")

        # build mixed precision scaler
        if cfg.TRAIN.amp:
            assert not isinstance(model, DataParallel), \
                "We do not support mixed precision training with DataParallel currently"
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def run(self):
        # the whole process for training
        for ep in range(self._start_epoch, self.cfg.TRAIN.epochs):
            self._epoch = ep

            # train stage
            self.train()
            synchronize()

            # validate stage
            if (ep + 1) % self.cfg.TRAIN.val_freq == 0 or (ep + 1) == self.cfg.TRAIN.epochs:
                
                # print the learning rete
                print("=== The lr of optimizer is {} ===".format(self.optimizer.param_groups[0]['lr']))

                # print the results
                if "val_dataset" in self.cfg.TRAIN:
                    self.save_model()
                    self.val()
                else:
                    self.save_model()

            # update learning rate
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, list):
                    for scheduler in self.lr_scheduler:
                        scheduler.step()
                elif isinstance(self.lr_scheduler, dict):
                    for key in self.lr_scheduler.keys():
                        self.lr_scheduler[key].step()
                else:
                    self.lr_scheduler.step()

            # synchronize distributed processes
            synchronize()
            torch.cuda.empty_cache()

    def train(self):
        # set training model
        if isinstance(self.model, list):
            for model in self.model:
                model.train()
        elif isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        # set training loader
        if isinstance(self.train_loader, list):
            for loader in self.train_loader:
                loader.new_epoch(self._epoch)
        else:
            self.train_loader.new_epoch(self._epoch)

        # update train progress
        self.train_progress.reset(prefix="Epoch: [{}]".format(self._epoch))

        # one loop for training
        end = time.time()
        for iter in range(self.cfg.TRAIN.iters):

            if isinstance(self.train_loader, list):
                batch = [loader.next() for loader in self.train_loader]
            else:
                batch = self.train_loader.next()

            if self.scaler is None:
                loss = self.train_step(iter, batch)
                if (loss > 0):
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                with autocast():
                    loss = self.train_step(iter, batch)
                if (loss > 0):
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)

            if self.scaler is not None:
                self.scaler.update()

            self.train_progress.update({"Time": time.time() - end})
            end = time.time()

            if iter % self.print_freq == 0:
                self.train_progress.display(iter)

    def train_step(self, iter, batch):
        # need to be re-written case by case
        assert not isinstance(
            self.model, list
        ), "please re-write 'train_step()' to support list of models"

        data = batch_processor(batch)

        inputs = torch.cat(data["image"], dim=0)
        targets = {
            "label": torch.cat(data["label"], dim=0),
            "views": torch.cat(data["views"], dim=0),
            "types": torch.cat(data["types"], dim=0),
            }

        results = self.model(inputs) # [B T H W]

        total_loss = 0
        for key in self.criterions.keys():
            loss, loss_info = self.criterions[key](results, targets)
            total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
            self.train_progress.update(loss_info)

        return total_loss

    def val(self):
        if not isinstance(self.model, list):
            model_list = [self.model]
        else:
            model_list = self.model

        for idx in range(len(model_list)):
            if len(model_list) > 1:
                print("==> Val on the no.{} model".format(idx))

            test_gait(
                self.cfg,
                model_list[idx],
                self.val_loader[0],
                self.val_set[0],
                self.cfg.TRAIN.val_dataset,
                self._rank,
                self._epoch,
                print_freq=self.print_freq,
            )

    def resume(self, path):
        # resume from a training checkpoint (not source pretrain)
        self.load_model(path)
        synchronize()

    def save_model(self):
        if self._rank == 0:
            # only on cuda:0
            if isinstance(self.model, list):
                state_dict = {}
                state_dict["epoch"] = self._epoch + 1
                for idx, model in enumerate(self.model):
                    state_dict["state_dict_" + str(idx + 1)] = model.state_dict()
                save_checkpoint(state_dict, fpath=osp.join(self.cfg.work_dir, "checkpoint_"+str(self._epoch + 1)+".pth"))

            elif isinstance(self.model, dict):
                state_dict = {}
                state_dict["epoch"] = self._epoch + 1
                for key in self.model.keys():
                    state_dict["state_dict"] = self.model[key].state_dict()
                    save_checkpoint(state_dict, fpath=osp.join(self.cfg.work_dir, "checkpoint_"+str(self._epoch + 1)+key+".pth"))

            elif isinstance(self.model, nn.Module):
                state_dict = {}
                state_dict["epoch"] = self._epoch + 1
                state_dict["state_dict"] = self.model.state_dict()
                save_checkpoint(state_dict, fpath=osp.join(self.cfg.work_dir, "checkpoint_"+str(self._epoch + 1)+".pth"))

            else:
                assert "Unknown type of model for save_model()"

    def load_model(self, path):
        if isinstance(self.model, list):
            assert osp.isfile(path)
            state_dict = load_checkpoint(path)
            for idx, model in enumerate(self.model):
                copy_state_dict(state_dict["state_dict_" + str(idx + 1)], model)

        elif isinstance(self.model, dict):
            assert osp.isdir(path)
            for key in self.model.keys():
                state_dict = load_checkpoint(osp.join(path, "checkpoint_"+key+".pth"))
                copy_state_dict(state_dict["state_dict"], self.model[key])

        elif isinstance(self.model, nn.Module):
            assert osp.isfile(path)
            state_dict = load_checkpoint(path)
            copy_state_dict(state_dict["state_dict"], self.model)

        self._start_epoch = state_dict["epoch"]

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size
