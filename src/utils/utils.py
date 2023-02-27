import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

from src.utils import pylogger, rich_utils

from itertools import permutations
import numpy as np
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from typing import Tuple

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # execute the task
        try:

            # apply extra utilities
            extras(cfg)

            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:

            # save exception to `.log` file
            log.exception("")

            # when using hydra plugins like Optuna, you might want to disable raising exception
            # to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:

            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)
        
### Utils SEL

class SELLoss(_Loss):
    """Custom sound event localization (SEL) loss function, which returns the sum of the binary cross-entropy loss
    regarding the estimated number of sources at each time-step and the minimum direction-of-arrival mean squared error
    loss, calculated according to all possible combinations of active sources."""

    __constants__ = ['reduction']

    def __init__(self,
                 max_num_sources: int,
                 alpha: float = 1.0,
                 size_average=None,
                 reduce=None,
                 reduction='mean') -> None:
        super(SELLoss, self).__init__(size_average, reduce, reduction)

        if (alpha < 0) or (alpha > 1):
            assert ValueError('The weighting parameter must be a number between 0 and 1.')

        self.alpha = alpha
        self.permutations = torch.from_numpy(np.array(list(permutations(range(max_num_sources)))))
        self.num_permutations = self.permutations.shape[0]

    @staticmethod
    def compute_spherical_distance(y_pred: torch.Tensor,
                                   y_true: torch.Tensor) -> torch.Tensor:
        if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
            assert RuntimeError('Input tensors require a dimension of two.')

        sine_term = torch.sin(y_pred[:, 0]) * torch.sin(y_true[:, 0])
        cosine_term = torch.cos(y_pred[:, 0]) * torch.cos(y_true[:, 0]) * torch.cos(y_true[:, 1] - y_pred[:, 1])

        return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        source_activity_pred, direction_of_arrival_pred, _ = predictions
        source_activity_target, direction_of_arrival_target = targets

        source_activity_bce_loss = F.binary_cross_entropy_with_logits(source_activity_pred, source_activity_target)

        source_activity_mask = source_activity_target.bool()

        spherical_distance = self.compute_spherical_distance(
            direction_of_arrival_pred[source_activity_mask], direction_of_arrival_target[source_activity_mask])
        direction_of_arrival_loss = self.alpha * torch.mean(spherical_distance)

        loss = source_activity_bce_loss + direction_of_arrival_loss

        meta_data = {
            'source_activity_loss': source_activity_bce_loss,
            'direction_of_arrival_loss': direction_of_arrival_loss
        }

        return loss, meta_data

### Adrenaline utils 

def compute_angular_distance(x, y):
    """Computes the angle between two spherical direction-of-arrival points.

    :param x: single direction-of-arrival, where the first column is the azimuth and second column is elevation
    :param y: single or multiple DoAs, where the first column is the azimuth and second column is elevation
    :return: angular distance
    """
    if np.ndim(x) != 1:
        raise ValueError('First DoA must be a single value.')

    return np.arccos(np.sin(x[0]) * np.sin(y[0]) + np.cos(x[0]) * np.cos(y[0]) * np.cos(y[1] - x[1]))


def get_num_params(model):
    """Returns the number of trainable parameters of a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

### Pilot utils

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple


def compute_spherical_distance(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Computes the distance between two points (given as angles) on a sphere, as described in Eq. (6) in the paper.

    Args:
        y_pred (Tensor): Tensor of predicted azimuth and elevation angles.
        y_true (Tensor): Tensor of ground-truth azimuth and elevation angles.

    Returns:
        Tensor: Tensor of spherical distances.
    """
    if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
        assert RuntimeError('Input tensors require a dimension of two.')

    sine_term = torch.sin(y_pred[:, 0]) * torch.sin(y_true[:, 0])
    cosine_term = torch.cos(y_pred[:, 0]) * torch.cos(y_true[:, 0]) * torch.cos(y_true[:, 1] - y_pred[:, 1])

    return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))


def compute_kld_to_standard_norm(covariance_matrix: Tensor) -> Tensor:
    """Computes the Kullback-Leibler divergence between two multivariate Gaussian distributions with identical mean,
    where the second distribution has an identity covariance matrix.

    Args:
        covariance_matrix (Tensor): Covariance matrix of the first distribution.

    Returns:
        Tensor: Tensor of KLD values.
    """
    matrix_dim = covariance_matrix.shape[-1]

    covariance_trace = torch.diagonal(covariance_matrix, dim1=-2, dim2=-1).sum(-1)

    return 0.5 * (covariance_trace - matrix_dim - torch.logdet(covariance_matrix.contiguous()))


def psel_loss(predictions: Tuple[Tensor, Tensor, Tensor],
              targets: Tuple[Tensor, Tensor],
              alpha: float = 1.,
              beta: float = 1.) -> Tensor:
    """Returns the probabilistic sound event localization loss, as described in Eq. (5) in the paper.

    Args:
        predictions (tuple): Predicted source activity, direction-of-arrival and posterior covariance matrix.
        targets (Tensor): Ground-truth source activity and direction-of-arrival.
        alpha (float): Weighting factor for direction-of-arrival loss component.
        beta (float): Weighting factor for KLD loss component.

    Returns:
        Tensor: Scalar probabilistic SEL loss value.
    """
    source_activity, posterior_mean, posterior_covariance = predictions
    source_activity_target, direction_of_arrival_target = targets

    source_activity_loss = F.binary_cross_entropy(source_activity, source_activity_target)
    source_activity_mask = source_activity_target.bool()

    spherical_distance = compute_spherical_distance(posterior_mean[source_activity_mask],
                                                    direction_of_arrival_target[source_activity_mask])
    direction_of_arrival_loss = torch.mean(spherical_distance)

    kld_loss = compute_kld_to_standard_norm(posterior_covariance)
    kld_loss = torch.mean(kld_loss)
    
    meta_data = {
            'source_activity_loss': source_activity_loss,
            'direction_of_arrival_loss': direction_of_arrival_loss,
            'kld_loss': kld_loss
        }

    return source_activity_loss + alpha * direction_of_arrival_loss + beta * kld_loss, meta_data