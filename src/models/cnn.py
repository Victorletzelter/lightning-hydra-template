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
        super(CNN, self).__init__(dataset_path, cv_fold_idx, hparams)

        num_steps_per_chunk = int(2 * hparams['chunk_length'] / hparams['frame_length'])
        
        print("ICI !!!")
        print(num_steps_per_chunk)
        
        self.feature_extraction = FeatureExtraction(num_steps_per_chunk=num_steps_per_chunk,
                                                    num_fft_bins=hparams['num_fft_bins'],
                                                    dropout_rate=hparams['dropout_rate'])

        feature_dim = int(hparams['num_fft_bins'] / 4)

        self.localization_output = LocalizationOutput(input_dim = feature_dim, max_num_sources = hparams['max_num_sources'])

    def get_loss_function(self) -> nn.Module:
        return SELLoss(self.hparams['max_num_sources'], alpha=self.hparams['alpha'])

    def forward(self,
                audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        
        print("audio features shape")
        print(audio_features.shape)
        
        extracted_features = self.feature_extraction(audio_features)
        
        print("extracted features shape")
        print(extracted_features.shape)

        source_activity_output, direction_of_arrival_output = self.localization_output(extracted_features)
        meta_data = {}

        return source_activity_output, direction_of_arrival_output, meta_data
