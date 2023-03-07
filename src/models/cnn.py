from argparse import Namespace
from .modules import AbstractLocalizationModule, FeatureExtraction, LocalizationOutput
import torch
import torch.nn as nn
from typing import Tuple
from src.utils import utils
from src.utils.utils import SELLoss

class CNN(AbstractLocalizationModule):
    def __init__(self,
                 dataset_path: str,
                 cv_fold_idx: int,
                 hparams: dict) -> None:
        """CNN Model for processing audio data. 

        Args:
            dataset_path (str): Path to the dataset folder.
            cv_fold_idx (int): Index associated with the cross-validation fold (1,2 or 3).
            hparams (dict): Dictionnary of hyperparameters with keys:
              name (str): Name of the model
              max_num_sources (int): Maximum number of sources (related the output shape).
              sequence_duration (int): Duration (s) of the audio samples.
              num_fft_bins (int): Number of frequencies in STFT computation.
              frame_length (int): Length (s) of the analysis window (Default to hann) used for FFT computation. 
              chunk_length (int): Duration (s) of the non-overlapping "chunks".
              dropout_rate (float): Dropout rate. 
              learning_rate (float): Initial learning rate to use in the optimization process. 
              num_epochs_warmup (int): Number of epochs from which the learning rate decay is enabled (see configure_optimizers in the AbstractLocalizationModule).
              alpha (float): Load of the direction_of_arrival_loss along with the source_activity_loss in the SEL (Sound Event Localization) Loss. 
              gradient_clip_val (float): 1.0.
              results_dir: /root/workspace/lightning-hydra-template/results.
        """
        super(CNN, self).__init__(dataset_path, cv_fold_idx, hparams)

        num_steps_per_chunk = int(2 * hparams['chunk_length'] / hparams['frame_length'])
        
        self.feature_extraction = FeatureExtraction(num_steps_per_chunk=num_steps_per_chunk,
                                                    num_fft_bins=hparams['num_fft_bins'],
                                                    dropout_rate=hparams['dropout_rate'])

        feature_dim = int(hparams['num_fft_bins'] / 4)

        self.localization_output = LocalizationOutput(input_dim = feature_dim, max_num_sources = hparams['max_num_sources'])

    def get_loss_function(self) -> nn.Module:
        return SELLoss(self.hparams['max_num_sources'], alpha=self.hparams['alpha'])

    def forward(self,
                audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        
        extracted_features = self.feature_extraction(audio_features)

        source_activity_output, direction_of_arrival_output = self.localization_output(extracted_features)
        meta_data = {}

        return source_activity_output, direction_of_arrival_output, meta_data
