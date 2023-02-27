### Adrenaline metrics

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from utils import compute_angular_distance


def frame_recall(predictions: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
    """Frame-recall metric, describing the percentage of frames where the number of predicted sources matches the number
    of sources provided in the ground-truth data. For additional information, refer to e.g.

        Adavanne et al.: "A multi-room reverberant dataset for sound event localization and detection" (2019)

    :param predictions: predicted source activities and doas
    :param targets: ground-truth source activities and doas
    :return: frame recall
    """
    predicted_source_activity = predictions[0].cpu()
    target_source_activity = targets[0].cpu()

    predicted_num_active_sources = torch.sum(torch.sigmoid(predicted_source_activity) > 0.5, dim=-1)
    target_num_active_sources = torch.sum(target_source_activity, dim=-1)

    frame_recall = torch.mean((predicted_num_active_sources == target_num_active_sources).float())

    return frame_recall


def doa_error(predictions: torch.Tensor,
              targets: torch.Tensor) -> torch.Tensor:
    batch_size, num_time_steps, _ = predictions[0].shape

    doa_error_matrix = np.zeros((batch_size, num_time_steps))

    for batch_idx in range(batch_size):
        for step_idx in range(num_time_steps):
            predicted_source_activity = (torch.sigmoid(predictions[0][batch_idx, step_idx, :]) > 0.5).detach().cpu().numpy()
            predicted_direction_of_arrival = predictions[1][batch_idx, step_idx, :, :].detach().cpu().numpy()
            target_source_activity = targets[0][batch_idx, step_idx, :].bool().detach().cpu().numpy()
            target_direction_of_arrival = targets[1][batch_idx, step_idx, :, :].detach().cpu().numpy()

            predicted_sources = predicted_direction_of_arrival[predicted_source_activity, :]
            num_predicted_sources = predicted_sources.shape[0]
            target_sources = target_direction_of_arrival[target_source_activity, :]
            num_target_sources = target_sources.shape[0]

            if (num_predicted_sources > 0) and (num_target_sources > 0):
                cost_matrix = np.zeros((num_predicted_sources, num_target_sources))

                for pred_idx in range(num_predicted_sources):
                    for target_idx in range(num_target_sources):
                        cost_matrix[pred_idx, target_idx] = compute_angular_distance(
                            predicted_sources[pred_idx, :], target_sources[target_idx, :])

                row_idx, col_idx = linear_sum_assignment(cost_matrix)
                doa_error_matrix[batch_idx, step_idx] = np.rad2deg(cost_matrix[row_idx, col_idx].mean())
            else:
                doa_error_matrix[batch_idx, step_idx] = np.nan

    return torch.tensor(np.nanmean(doa_error_matrix, dtype=np.float32))

### Pilot metrics 

from .utils import compute_spherical_distance
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import Tensor
from torchmetrics import Metric


class FrameRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        """
        Args:
            dist_sync_on_step:
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, source_activity_prediction: Tensor, source_activity_target: Tensor) -> None:
        """
        Args:
            source_activity_prediction (Tensor):
            source_activity_target (Tensor):
        """
        assert source_activity_prediction.shape == source_activity_target.shape

        num_active_sources_prediction = torch.sum(source_activity_prediction > 0.5, dim=1)
        num_active_sources_target = torch.sum(source_activity_target, dim=1)

        self.correct += torch.sum(num_active_sources_prediction == num_active_sources_target)
        self.total += source_activity_target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total

    @property
    def is_differentiable(self) -> bool:
        return False


class DOAError(Metric):
    def __init__(self, dist_sync_on_step=False):
        """
        Args:
            dist_sync_on_step:
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('sum_doa_error', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               source_activity_prediction: Tensor,
               direction_of_arrival_prediction: Tensor,
               source_activity_target: Tensor,
               direction_of_arrival_target: Tensor,
               ) -> None:
        """
        Args:
            source_activity_prediction (Tensor):
            direction_of_arrival_prediction (Tensor):
            source_activity_target (Tensor):
            direction_of_arrival_target (Tensor):
        """
        batch_size, max_num_sources, num_steps = source_activity_prediction.shape

        for batch_idx in range(batch_size):
            for step_idx in range(num_steps):
                active_sources_prediction = source_activity_prediction[batch_idx, :, step_idx] > 0.5
                active_sources_target = source_activity_target.bool()[batch_idx, :, step_idx]
                num_predicted_sources = active_sources_prediction.sum()
                num_target_sources = active_sources_target.sum()

                if (num_predicted_sources > 0) and (num_target_sources > 0):
                    predicted_sources = direction_of_arrival_prediction[batch_idx, active_sources_prediction, step_idx, :]
                    target_sources = direction_of_arrival_target[batch_idx, active_sources_target, step_idx, :]

                    cost_matrix = np.zeros((num_predicted_sources, num_target_sources))

                    for i in range(num_predicted_sources):
                        for j in range(num_target_sources):
                            cost_matrix[i, j] = compute_spherical_distance(predicted_sources[i, :].unsqueeze(0),
                                                                           target_sources[j, :].unsqueeze(0)).cpu().numpy()

                    row_idx, col_idx = linear_sum_assignment(cost_matrix)

                    self.sum_doa_error += np.rad2deg(cost_matrix[row_idx, col_idx].mean())
                    self.total += 1

    def compute(self) -> Tensor:
        return self.sum_doa_error / self.total if self.total > 0 else 180.

    @property
    def is_differentiable(self) -> bool:
        return False

