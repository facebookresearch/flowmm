"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import resource
from pathlib import Path
from typing import List

import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from diffcsp.common.utils import log_hyperparameters
from flowmm.model.eval_utils import register_omega_conf_resolvers
from rfm_docking.docking_model_pl import DockingRFMLitModule

# https://github.com/Project-MONAI/MONAI/issues/701#issuecomment-767330310
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.multiprocessing.set_sharing_strategy("file_system")


try:
    WANDB_MODE = os.environ["WANDB_MODE"]
except KeyError:
    WANDB_MODE = ""


register_omega_conf_resolvers()


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if (WANDB_MODE.lower() != "disabled") and ("lr_monitor" in cfg.logging):
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    if "every_n_epochs_checkpoint" in cfg.train:
        hydra.utils.log.info(
            f"Adding callback <ModelCheckpoint> for every {cfg.train.every_n_epochs_checkpoint.every_n_epochs} epochs"
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath="every_n_epochs",
                every_n_epochs=cfg.train.every_n_epochs_checkpoint.every_n_epochs,
                save_top_k=cfg.train.every_n_epochs_checkpoint.save_top_k,
                verbose=cfg.train.every_n_epochs_checkpoint.verbose,
                save_last=cfg.train.every_n_epochs_checkpoint.save_last,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    # hydra_dir = Path(HydraConfig.get().run.dir)
    hydra_dir = Path.cwd()
    hydra.utils.log.info(f"Hydra Directory is {hydra_dir.resolve()}")

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    get_model = DockingRFMLitModule
    hydra.utils.log.info(f"Instantiating <{get_model}>")
    model = get_model(cfg)

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    do_wandb_log = (WANDB_MODE.lower() != "disabled") and ("wandb" in cfg.logging)
    if do_wandb_log:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # Load checkpoint (if exist)
    ckpts = list(hydra_dir.glob("*.ckpt"))
    if len(ckpts) > 0:
        ckpt_epochs = np.array(
            [int(ckpt.parts[-1].split("-")[0].split("=")[1]) for ckpt in ckpts]
        )
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
    else:
        ckpt = None

    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        # default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        # progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        resume_from_checkpoint=ckpt,
        **cfg.train.pl_trainer,
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    if do_wandb_log:
        hydra.utils.log.info(
            "W&B is no longer watching <{cfg.logging.wandb_watch.log}>!"
        )
        wandb_logger.experiment.unwatch(model)

    hydra.utils.log.info("Starting testing!")
    ckpt_path = "last" if cfg.train.pl_trainer.fast_dev_run else "best"
    # trainer.test(datamodule=datamodule, ckpt_path=ckpt_path)
    traj = trainer.predict(
        dataloaders=datamodule.test_dataloader(), ckpt_path=ckpt_path
    )

    torch.save(traj, "traj.pt")

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(
    config_path="conf",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
