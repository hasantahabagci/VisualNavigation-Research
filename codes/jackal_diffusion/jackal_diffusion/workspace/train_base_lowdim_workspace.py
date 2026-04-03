from __future__ import annotations

import copy
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from jackal_diffusion.common import JsonLogger, TopKCheckpointManager, dict_apply, optimizer_to
from jackal_diffusion.dataset.base_dataset import BaseLowdimDataset
from jackal_diffusion.env_runner.base_lowdim_runner import BaseLowdimRunner
from jackal_diffusion.policy.base_lowdim_policy import BaseLowdimPolicy
from jackal_diffusion.workspace.base_workspace import BaseWorkspace


class TrainBaseLowdimWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch", "train_losses", "val_losses", "last_runner_log")

    def __init__(self, cfg: OmegaConf, output_dir=None) -> None:
        super().__init__(cfg, output_dir=output_dir)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: BaseLowdimPolicy = hydra.utils.instantiate(cfg.policy)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.last_runner_log: dict = {}

    def run(self) -> dict:
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path, map_location="cpu")

        dataset: BaseLowdimDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)

        env_runner: BaseLowdimRunner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir,
        )
        assert isinstance(env_runner, BaseLowdimRunner)

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)
        os.makedirs(self.output_dir, exist_ok=True)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        train_info = dict(getattr(dataset, "describe", lambda: {})())
        if train_info:
            print(f"[Data]     {train_info}")

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log = {
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                }
                train_loss = self._run_train_epoch(train_dataloader, cfg, device)
                self.train_losses.append(train_loss)
                step_log["train_loss"] = train_loss

                if (self.epoch % cfg.training.val_every) == 0:
                    val_loss = self._run_val_epoch(val_dataloader, cfg, device)
                    self.val_losses.append(val_loss)
                    step_log["val_loss"] = val_loss

                if (self.epoch % cfg.training.rollout_every) == 0:
                    self.model.eval()
                    self.last_runner_log = env_runner.run(self.model)
                    step_log.update(self.last_runner_log)
                    self.model.train()

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    monitor_key = cfg.checkpoint.topk.monitor_key
                    if monitor_key in step_log:
                        topk_ckpt_path = topk_manager.get_ckpt_path(step_log)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

                if self._should_early_stop(step_log, cfg):
                    break
        return {
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
            "runner_log": dict(self.last_runner_log),
        }

    def _run_train_epoch(self, dataloader, cfg, device) -> float:
        self.model.train()
        losses = []
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(dataloader):
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            raw_loss = self.model.compute_loss(batch)
            loss = raw_loss / cfg.training.gradient_accumulate_every
            loss.backward()

            if ((batch_idx + 1) % cfg.training.gradient_accumulate_every) == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            losses.append(float(raw_loss.item()))
            if (
                cfg.training.max_train_steps is not None
                and batch_idx >= (cfg.training.max_train_steps - 1)
            ):
                break

        if losses and (len(losses) % cfg.training.gradient_accumulate_every) != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return float(np.mean(losses)) if losses else 0.0

    def _run_val_epoch(self, dataloader, cfg, device) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss = self.model.compute_loss(batch)
                losses.append(float(loss.item()))
                if (
                    cfg.training.max_val_steps is not None
                    and batch_idx >= (cfg.training.max_val_steps - 1)
                ):
                    break
        return float(np.mean(losses)) if losses else 0.0

    def _should_early_stop(self, step_log: dict, cfg: OmegaConf) -> bool:
        threshold = cfg.training.early_stop_loss
        if threshold is None:
            return False
        val_loss = step_log.get("val_loss")
        if val_loss is None:
            return False
        return val_loss < threshold
