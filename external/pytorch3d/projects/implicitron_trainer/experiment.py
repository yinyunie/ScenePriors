#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

""""
This file is the entry point for launching experiments with Implicitron.

Main functions
---------------
- `run_training` is the wrapper for the train, val, test loops
    and checkpointing
- `trainvalidate` is the inner loop which runs the model forward/backward
    pass, visualizations and metric printing

Launch Training
---------------
Experiment config .yaml files are located in the
`projects/implicitron_trainer/configs` folder. To launch
an experiment, specify the name of the file. Specific config values can
also be overridden from the command line, for example:

```
./experiment.py --config-name base_config.yaml override.param.one=42 override.param.two=84
```

To run an experiment on a specific GPU, specify the `gpu_idx` key
in the config file / CLI. To run on a different device, specify the
device in `run_training`.

Outputs
--------
The outputs of the experiment are saved and logged in multiple ways:
  - Checkpoints:
        Model, optimizer and stats are stored in the directory
        named by the `exp_dir` key from the config file / CLI parameters.
  - Stats
        Stats are logged and plotted to the file "train_stats.pdf" in the
        same directory. The stats are also saved as part of the checkpoint file.
  - Visualizations
        Prredictions are plotted to a visdom server running at the
        port specified by the `visdom_server` and `visdom_port` keys in the
        config file.

"""
import copy
import json
import logging
import os
import random
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import hydra
import lpips
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from packaging import version
from pytorch3d.implicitron.dataset import utils as ds_utils
from pytorch3d.implicitron.dataset.data_loader_map_provider import DataLoaderMap
from pytorch3d.implicitron.dataset.data_source import ImplicitronDataSource, Task
from pytorch3d.implicitron.dataset.dataset_map_provider import DatasetMap
from pytorch3d.implicitron.evaluation import evaluate_new_view_synthesis as evaluate
from pytorch3d.implicitron.models.generic_model import EvaluationMode, GenericModel
from pytorch3d.implicitron.models.renderer.multipass_ea import (
    MultiPassEmissionAbsorptionRenderer,
)
from pytorch3d.implicitron.models.renderer.ray_sampler import AdaptiveRaySampler
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.implicitron.tools.config import (
    expand_args_fields,
    remove_unused_components,
)
from pytorch3d.implicitron.tools.stats import Stats
from pytorch3d.renderer.cameras import CamerasBase

from .impl.experiment_config import ExperimentConfig
from .impl.optimization import init_optimizer

logger = logging.getLogger(__name__)

if version.parse(hydra.__version__) < version.Version("1.1"):
    raise ValueError(
        f"Hydra version {hydra.__version__} is too old."
        " (Implicitron requires version 1.1 or later.)"
    )

try:
    # only makes sense in FAIR cluster
    import pytorch3d.implicitron.fair_cluster.slurm  # noqa: F401
except ModuleNotFoundError:
    pass


def init_model(
    cfg: DictConfig,
    force_load: bool = False,
    clear_stats: bool = False,
    load_model_only: bool = False,
    accelerator: Accelerator = None,
) -> Tuple[GenericModel, Stats, Optional[Dict[str, Any]]]:
    """
    Returns an instance of `GenericModel`.

    If `cfg.resume` is set or `force_load` is true,
    attempts to load the last checkpoint from `cfg.exp_dir`. Failure to do so
    will return the model with initial weights, unless `force_load` is passed,
    in which case a FileNotFoundError is raised.

    Args:
        force_load: If true, force load model from checkpoint even if
            cfg.resume is false.
        clear_stats: If true, clear the stats object loaded from checkpoint
        load_model_only: If true, load only the model weights from checkpoint
            and do not load the state of the optimizer and stats.

    Returns:
        model: The model with optionally loaded weights from checkpoint
        stats: The stats structure (optionally loaded from checkpoint)
        optimizer_state: The optimizer state dict containing
            `state` and `param_groups` keys (optionally loaded from checkpoint)

    Raise:
        FileNotFoundError if `force_load` is passed but checkpoint is not found.
    """

    # Initialize the model
    if cfg.architecture == "generic":
        model = GenericModel(**cfg.generic_model_args)
    else:
        raise ValueError(f"No such arch {cfg.architecture}.")

    # Determine the network outputs that should be logged
    if hasattr(model, "log_vars"):
        log_vars = copy.deepcopy(list(model.log_vars))
    else:
        log_vars = ["objective"]

    visdom_env_charts = vis_utils.get_visdom_env(cfg) + "_charts"

    # Init the stats struct
    stats = Stats(
        log_vars,
        visdom_env=visdom_env_charts,
        verbose=False,
        visdom_server=cfg.visdom_server,
        visdom_port=cfg.visdom_port,
    )

    # Retrieve the last checkpoint
    if cfg.resume_epoch > 0:
        model_path = model_io.get_checkpoint(cfg.exp_dir, cfg.resume_epoch)
    else:
        model_path = model_io.find_last_checkpoint(cfg.exp_dir)

    optimizer_state = None
    if model_path is not None:
        logger.info("found previous model %s" % model_path)
        if force_load or cfg.resume:
            logger.info("   -> resuming")

            map_location = None
            if not accelerator.is_local_main_process:
                map_location = {
                    "cuda:%d" % 0: "cuda:%d" % accelerator.local_process_index
                }
            if load_model_only:
                model_state_dict = torch.load(
                    model_io.get_model_path(model_path), map_location=map_location
                )
                stats_load, optimizer_state = None, None
            else:
                model_state_dict, stats_load, optimizer_state = model_io.load_model(
                    model_path, map_location=map_location
                )

                # Determine if stats should be reset
                if not clear_stats:
                    if stats_load is None:
                        logger.info("\n\n\n\nCORRUPT STATS -> clearing stats\n\n\n\n")
                        last_epoch = model_io.parse_epoch_from_model_path(model_path)
                        logger.info(f"Estimated resume epoch = {last_epoch}")

                        # Reset the stats struct
                        for _ in range(last_epoch + 1):
                            stats.new_epoch()
                        assert last_epoch == stats.epoch
                    else:
                        stats = stats_load

                    # Update stats properties incase it was reset on load
                    stats.visdom_env = visdom_env_charts
                    stats.visdom_server = cfg.visdom_server
                    stats.visdom_port = cfg.visdom_port
                    stats.plot_file = os.path.join(cfg.exp_dir, "train_stats.pdf")
                    stats.synchronize_logged_vars(log_vars)
                else:
                    logger.info("   -> clearing stats")

            try:
                # TODO: fix on creation of the buffers
                # after the hack above, this will not pass in most cases
                # ... but this is fine for now
                model.load_state_dict(model_state_dict, strict=True)
            except RuntimeError as e:
                logger.error(e)
                logger.info("Cant load state dict in strict mode! -> trying non-strict")
                model.load_state_dict(model_state_dict, strict=False)
            model.log_vars = log_vars
        else:
            logger.info("   -> but not resuming -> starting from scratch")
    elif force_load:
        raise FileNotFoundError(f"Cannot find a checkpoint in {cfg.exp_dir}!")

    return model, stats, optimizer_state


def trainvalidate(
    model,
    stats,
    epoch,
    loader,
    optimizer,
    validation: bool,
    bp_var: str = "objective",
    metric_print_interval: int = 5,
    visualize_interval: int = 100,
    visdom_env_root: str = "trainvalidate",
    clip_grad: float = 0.0,
    device: str = "cuda:0",
    accelerator: Accelerator = None,
    **kwargs,
) -> None:
    """
    This is the main loop for training and evaluation including:
    model forward pass, loss computation, backward pass and visualization.

    Args:
        model: The model module optionally loaded from checkpoint
        stats: The stats struct, also optionally loaded from checkpoint
        epoch: The index of the current epoch
        loader: The dataloader to use for the loop
        optimizer: The optimizer module optionally loaded from checkpoint
        validation: If true, run the loop with the model in eval mode
            and skip the backward pass
        bp_var: The name of the key in the model output `preds` dict which
            should be used as the loss for the backward pass.
        metric_print_interval: The batch interval at which the stats should be
            logged.
        visualize_interval: The batch interval at which the visualizations
            should be plotted
        visdom_env_root: The name of the visdom environment to use for plotting
        clip_grad: Optionally clip the gradient norms.
            If set to a value <=0.0, no clipping
        device: The device on which to run the model.

    Returns:
        None
    """

    if validation:
        model.eval()
        trainmode = "val"
    else:
        model.train()
        trainmode = "train"

    t_start = time.time()

    # get the visdom env name
    visdom_env_imgs = visdom_env_root + "_images_" + trainmode
    viz = vis_utils.get_visdom_connection(
        server=stats.visdom_server,
        port=stats.visdom_port,
    )

    # Iterate through the batches
    n_batches = len(loader)
    for it, net_input in enumerate(loader):
        last_iter = it == n_batches - 1

        # move to gpu where possible (in place)
        net_input = net_input.to(accelerator.device)

        # run the forward pass
        if not validation:
            optimizer.zero_grad()
            preds = model(**{**net_input, "evaluation_mode": EvaluationMode.TRAINING})
        else:
            with torch.no_grad():
                preds = model(
                    **{**net_input, "evaluation_mode": EvaluationMode.EVALUATION}
                )

        # make sure we dont overwrite something
        assert all(k not in preds for k in net_input.keys())
        # merge everything into one big dict
        preds.update(net_input)

        # update the stats logger
        stats.update(preds, time_start=t_start, stat_set=trainmode)
        assert stats.it[trainmode] == it, "inconsistent stat iteration number!"

        # print textual status update
        if it % metric_print_interval == 0 or last_iter:
            stats.print(stat_set=trainmode, max_it=n_batches)

        # visualize results
        if (
            accelerator.is_local_main_process
            and visualize_interval > 0
            and it % visualize_interval == 0
        ):
            prefix = f"e{stats.epoch}_it{stats.it[trainmode]}"

            model.visualize(
                viz,
                visdom_env_imgs,
                preds,
                prefix,
            )

        # optimizer step
        if not validation:
            loss = preds[bp_var]
            assert torch.isfinite(loss).all(), "Non-finite loss!"
            # backprop
            accelerator.backward(loss)
            if clip_grad > 0.0:
                # Optionally clip the gradient norms.
                total_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(), clip_grad
                )
                if total_norm > clip_grad:
                    logger.info(
                        f"Clipping gradient: {total_norm}"
                        + f" with coef {clip_grad / float(total_norm)}."
                    )

            optimizer.step()


def run_training(cfg: DictConfig) -> None:
    """
    Entry point to run the training and validation loops
    based on the specified config file.
    """

    # Initialize the accelerator
    accelerator = Accelerator(device_placement=False)
    logger.info(accelerator.state)

    device = accelerator.device
    logger.info(f"Running experiment on device: {device}")

    if accelerator.is_local_main_process:
        logger.info(OmegaConf.to_yaml(cfg))

    # set the debug mode
    if cfg.detect_anomaly:
        logger.info("Anomaly detection!")
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

    # create the output folder
    os.makedirs(cfg.exp_dir, exist_ok=True)
    _seed_all_random_engines(cfg.seed)
    remove_unused_components(cfg)

    # dump the exp config to the exp dir
    try:
        cfg_filename = os.path.join(cfg.exp_dir, "expconfig.yaml")
        OmegaConf.save(config=cfg, f=cfg_filename)
    except PermissionError:
        warnings.warn("Cant dump config due to insufficient permissions!")

    # setup datasets
    datasource = ImplicitronDataSource(**cfg.data_source_args)
    datasets, dataloaders = datasource.get_datasets_and_dataloaders()
    task = datasource.get_task()
    all_train_cameras = datasource.get_all_train_cameras()

    # init the model
    model, stats, optimizer_state = init_model(cfg, accelerator=accelerator)
    start_epoch = stats.epoch + 1

    # move model to gpu
    model.to(accelerator.device)

    # only run evaluation on the test dataloader
    if cfg.eval_only:
        _eval_and_dump(
            cfg,
            task,
            all_train_cameras,
            datasets,
            dataloaders,
            model,
            stats,
            device=device,
            accelerator=accelerator,
        )
        return

    # init the optimizer
    optimizer, scheduler = init_optimizer(
        model,
        optimizer_state=optimizer_state,
        last_epoch=start_epoch,
        **cfg.solver_args,
    )

    # check the scheduler and stats have been initialized correctly
    assert scheduler.last_epoch == stats.epoch + 1
    assert scheduler.last_epoch == start_epoch

    # Wrap all modules in the distributed library
    # Note: we don't pass the scheduler to prepare as it
    # doesn't need to be stepped at each optimizer step
    (
        model,
        optimizer,
        train_loader,
        val_loader,
    ) = accelerator.prepare(model, optimizer, dataloaders.train, dataloaders.val)

    past_scheduler_lrs = []
    # loop through epochs
    for epoch in range(start_epoch, cfg.solver_args.max_epochs):
        # automatic new_epoch and plotting of stats at every epoch start
        with stats:

            # Make sure to re-seed random generators to ensure reproducibility
            # even after restart.
            _seed_all_random_engines(cfg.seed + epoch)

            cur_lr = float(scheduler.get_last_lr()[-1])
            logger.info(f"scheduler lr = {cur_lr:1.2e}")
            past_scheduler_lrs.append(cur_lr)

            # train loop
            trainvalidate(
                model,
                stats,
                epoch,
                train_loader,
                optimizer,
                False,
                visdom_env_root=vis_utils.get_visdom_env(cfg),
                device=device,
                accelerator=accelerator,
                **cfg,
            )

            # val loop (optional)
            if val_loader is not None and epoch % cfg.validation_interval == 0:
                trainvalidate(
                    model,
                    stats,
                    epoch,
                    val_loader,
                    optimizer,
                    True,
                    visdom_env_root=vis_utils.get_visdom_env(cfg),
                    device=device,
                    accelerator=accelerator,
                    **cfg,
                )

            # eval loop (optional)
            if (
                dataloaders.test is not None
                and cfg.test_interval > 0
                and epoch % cfg.test_interval == 0
            ):
                _run_eval(
                    model,
                    all_train_cameras,
                    dataloaders.test,
                    task,
                    camera_difficulty_bin_breaks=cfg.camera_difficulty_bin_breaks,
                    device=device,
                    accelerator=accelerator,
                )

            assert stats.epoch == epoch, "inconsistent stats!"

            # delete previous models if required
            # save model only on the main process
            if cfg.store_checkpoints and accelerator.is_local_main_process:
                if cfg.store_checkpoints_purge > 0:
                    for prev_epoch in range(epoch - cfg.store_checkpoints_purge):
                        model_io.purge_epoch(cfg.exp_dir, prev_epoch)
                outfile = model_io.get_checkpoint(cfg.exp_dir, epoch)
                unwrapped_model = accelerator.unwrap_model(model)
                model_io.safe_save_model(
                    unwrapped_model, stats, outfile, optimizer=optimizer
                )

            scheduler.step()

            new_lr = float(scheduler.get_last_lr()[-1])
            if new_lr != cur_lr:
                logger.info(f"LR change! {cur_lr} -> {new_lr}")

    if cfg.test_when_finished:
        _eval_and_dump(
            cfg,
            task,
            all_train_cameras,
            datasets,
            dataloaders,
            model,
            stats,
            device=device,
        )


def _eval_and_dump(
    cfg,
    task: Task,
    all_train_cameras: Optional[CamerasBase],
    datasets: DatasetMap,
    dataloaders: DataLoaderMap,
    model,
    stats,
    device,
    accelerator: Accelerator = None,
) -> None:
    """
    Run the evaluation loop with the test data loader and
    save the predictions to the `exp_dir`.
    """

    dataloader = dataloaders.test

    if dataloader is None:
        raise ValueError('DataLoaderMap have to contain the "test" entry for eval!')

    results = _run_eval(
        model,
        all_train_cameras,
        dataloader,
        task,
        camera_difficulty_bin_breaks=cfg.camera_difficulty_bin_breaks,
        device=device,
        accelerator=accelerator,
    )

    # add the evaluation epoch to the results
    for r in results:
        r["eval_epoch"] = int(stats.epoch)

    logger.info("Evaluation results")
    evaluate.pretty_print_nvs_metrics(results)

    with open(os.path.join(cfg.exp_dir, "results_test.json"), "w") as f:
        json.dump(results, f)


def _get_eval_frame_data(frame_data):
    """
    Masks the unknown image data to make sure we cannot use it at model evaluation time.
    """
    frame_data_for_eval = copy.deepcopy(frame_data)
    is_known = ds_utils.is_known_frame(frame_data.frame_type).type_as(
        frame_data.image_rgb
    )[:, None, None, None]
    for k in ("image_rgb", "depth_map", "fg_probability", "mask_crop"):
        value_masked = getattr(frame_data_for_eval, k).clone() * is_known
        setattr(frame_data_for_eval, k, value_masked)
    return frame_data_for_eval


def _run_eval(
    model,
    all_train_cameras,
    loader,
    task: Task,
    camera_difficulty_bin_breaks: Tuple[float, float],
    device,
    accelerator: Accelerator = None,
):
    """
    Run the evaluation loop on the test dataloader
    """
    lpips_model = lpips.LPIPS(net="vgg")
    lpips_model = lpips_model.to(accelerator.device)

    model.eval()

    per_batch_eval_results = []
    logger.info("Evaluating model ...")
    for frame_data in tqdm.tqdm(loader):
        frame_data = frame_data.to(accelerator.device)

        # mask out the unknown images so that the model does not see them
        frame_data_for_eval = _get_eval_frame_data(frame_data)

        with torch.no_grad():
            preds = model(
                **{**frame_data_for_eval, "evaluation_mode": EvaluationMode.EVALUATION}
            )

            # TODO: Cannot use accelerate gather for two reasons:.
            # (1) TypeError: Can't apply _gpu_gather_one on object of type
            # <class 'pytorch3d.implicitron.models.base_model.ImplicitronRender'>,
            # only of nested list/tuple/dicts of objects that satisfy is_torch_tensor.
            # (2) Same error above but for frame_data which contains Cameras.

            implicitron_render = copy.deepcopy(preds["implicitron_render"])

            per_batch_eval_results.append(
                evaluate.eval_batch(
                    frame_data,
                    implicitron_render,
                    bg_color="black",
                    lpips_model=lpips_model,
                    source_cameras=all_train_cameras,
                )
            )

    _, category_result = evaluate.summarize_nvs_eval_results(
        per_batch_eval_results, task, camera_difficulty_bin_breaks
    )

    return category_result["results"]


def _seed_all_random_engines(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def _setup_envvars_for_cluster() -> bool:
    """
    Prepares to run on cluster if relevant.
    Returns whether FAIR cluster in use.
    """
    # TODO: How much of this is needed in general?

    try:
        import submitit
    except ImportError:
        return False

    try:
        # Only needed when launching on cluster with slurm and submitit
        job_env = submitit.JobEnvironment()
    except RuntimeError:
        return False

    os.environ["LOCAL_RANK"] = str(job_env.local_rank)
    os.environ["RANK"] = str(job_env.global_rank)
    os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "42918"
    logger.info(
        "Num tasks %s, global_rank %s"
        % (str(job_env.num_tasks), str(job_env.global_rank))
    )

    return True


expand_args_fields(ExperimentConfig)
cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="default_config", node=ExperimentConfig)


@hydra.main(config_path="./configs/", config_name="default_config")
def experiment(cfg: DictConfig) -> None:
    # CUDA_VISIBLE_DEVICES must have been set.

    if "CUDA_DEVICE_ORDER" not in os.environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if not _setup_envvars_for_cluster():
        logger.info("Running locally")

    # TODO: The following may be needed for hydra/submitit it to work
    expand_args_fields(GenericModel)
    expand_args_fields(AdaptiveRaySampler)
    expand_args_fields(MultiPassEmissionAbsorptionRenderer)
    expand_args_fields(ImplicitronDataSource)

    run_training(cfg)


if __name__ == "__main__":
    experiment()
