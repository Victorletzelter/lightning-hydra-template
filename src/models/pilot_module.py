import functools
import operator
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple
### Adapted 
import pytorch_lightning as ptl
import abc
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from src.data.data_handlers import TUTSoundEvents
import math
from src.metrics import frame_recall, doa_error
import numpy as np
import os
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import json

class AbstractLocalizationModule(ptl.LightningModule, abc.ABC):
    def __init__(self, dataset_path: str, cv_fold_idx: int, hparams):
        super(AbstractLocalizationModule, self).__init__()

        self.dataset_path = dataset_path
        self.cv_fold_idx = cv_fold_idx

        self._hparams = hparams
        self.max_num_sources = hparams['max_num_sources']
        
        if 'max_num_overlapping_sources_test' in hparams :
            self.max_num_overlapping_sources_test = hparams['max_num_overlapping_sources_test']
            
        else : 
            self.max_num_overlapping_sources_test = self.max_num_sources
         
        self.loss_function = self.get_loss_function()
        
    @property
    def hparams(self):
        return self._hparams

    @abc.abstractmethod
    def forward(self,
                audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_loss_function(self) -> nn.Module:
        raise NotImplementedError

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams['learning_rate'], weight_decay=0.0)

        lr_lambda = lambda epoch: self.hparams['learning_rate'] * np.minimum(
            (epoch + 1) ** -0.5, (epoch + 1) * (self.hparams['num_epochs_warmup'] ** -1.5)
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [scheduler]

    def training_step(self,
                      batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                      batch_idx: int) -> Dict:
        predictions, targets = self._process_batch(batch)

        loss, meta_data = self.loss_function(predictions, targets)

        output = {'loss': loss}
        self.log_dict(output)

        return output

    def validation_step(self,
                        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                        batch_idx: int) -> Dict:
        predictions, targets = self._process_batch(batch)

        loss, meta_data = self.loss_function(predictions, targets)

        output = {'val_loss': loss}
        self.log_dict(output)

        return output

    def validation_epoch_end(self,
                             outputs: list) -> None:
        average_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log_dict({'val_loss': average_loss, 'learning_rate': learning_rate})
        
        return average_loss

    def test_step(self,
                  batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                  batch_idx: int,
                  dataset_idx: int = 0) -> Dict:
        predictions, targets = self._process_batch(batch)

        output = {
            'test_frame_recall': frame_recall(predictions, targets), 'test_doa_error': doa_error(predictions, targets)
        }
        self.log_dict(output)

        return output
    
    def test_epoch_end(self,
                       outputs: List) -> None:
        dataset_name = os.path.split(self.dataset_path)[-1]
        model_name = '_'.join([self.hparams['name'], dataset_name, 'fold' + str(self.cv_fold_idx)])

        results = {
            'model': [], 'dataset': [], 'fold_idx': [], 'subset_idx': [], 'frame_recall': [], 'doa_error': []
        }
        
        #We check whether outputs if a list of List[Dict] (case with several subsets) or if it is a list of dicts (case of one subset)  
            
        print("test ici")
        print(len(outputs))
        print(outputs[0])    
               
        if len(outputs)>0 and isinstance(outputs[0],list) : 
            
            num_subsets = len(outputs)

            for subset_idx in range(num_subsets):
                frame_recall = torch.stack([x['test_frame_recall'] for x in outputs[subset_idx]]).detach().cpu().numpy()
                doa_error = torch.stack([x['test_doa_error'] for x in outputs[subset_idx]]).detach().cpu().numpy()

                num_sequences = len(frame_recall)

                for seq_idx in range(num_sequences):
                    results['model'].append(self.hparams['name'])
                    results['dataset'].append(dataset_name)
                    results['fold_idx'].append(self.cv_fold_idx)
                    results['subset_idx'].append(subset_idx)
                    results['frame_recall'].append(frame_recall[seq_idx])
                    results['doa_error'].append(doa_error[seq_idx])
                    
            data_frame = pd.DataFrame.from_dict(results)
            
            average_frame_recall = torch.tensor(data_frame['frame_recall'].mean(), dtype=torch.float32)
            average_doa_error = torch.tensor(data_frame['doa_error'].mean(), dtype=torch.float32)
            
            results_file = os.path.join(self.hparams['results_dir'], self.hparams['name'] + '_'
                                        + dataset_name + '_' + 'fold' + str(self.cv_fold_idx) + '_'
                                    'max_sources'+ str(self.max_num_sources) +  '_' +
                                    'num_dataloders'+ str(len(data_frame['frame_recall']))
                                    + '.json')

            if not os.path.isfile(results_file):
                data_frame.to_json(results_file)
                    
        elif len(outputs)>0 and isinstance(outputs[0],dict) : 
            
            frame_recall = torch.stack([x['test_frame_recall'] for x in outputs]).detach().cpu().numpy()
            doa_error = torch.stack([x['test_doa_error'] for x in outputs]).detach().cpu().numpy()

            num_sequences = len(frame_recall)
            
            _results = {'frame_recall' : [], 'doa_error' : []} 

            for seq_idx in range(num_sequences):
                _results['frame_recall'].append(float(frame_recall[seq_idx]))
                _results['doa_error'].append(float(doa_error[seq_idx]))
                
            data_frame = pd.DataFrame.from_dict(_results)
            
            average_frame_recall = torch.tensor(data_frame['frame_recall'].mean(), dtype=torch.float32)
            average_doa_error = torch.tensor(data_frame['doa_error'].mean(), dtype=torch.float32)
            
            _results['model']=self.hparams['name']
            _results['dataset']=dataset_name
            _results['fold_idx']=self.cv_fold_idx
            _results['average_frame_recall'] = float(average_frame_recall)
            _results['average_doa_error'] = float(average_doa_error)
            
            results_file = os.path.join(self.hparams['results_dir'], self.hparams['name'] + '_'
                                        + dataset_name + '_' + 'fold' + str(self.cv_fold_idx) + '_'
                                    'max_sources'+ str(self.max_num_sources) +  '_' +
                                    'num_test_samples'+ str(len(data_frame['frame_recall']))
                                    + '.json')
            
            if not os.path.isfile(results_file):
                with open(str(results_file),'w') as file : 
                    json.dump(_results, file)

        self.log_dict({'test_frame_recall': average_frame_recall, 'test_doa_error': average_doa_error})


    ### TO DELETE 
    
    # def test_epoch_end(self, 
    #                    outputs: List) -> None:
    #     dataset_name = os.path.split(self.dataset_path)[-1]
    #     model_name = '_'.join([self.hparams['name'], dataset_name, 'fold' + str(self.cv_fold_idx)])
    #     results_file = os.path.join(self.hparams['results_dir'], model_name + '.json')

    #     results = {
    #         'model': [], 'dataset': [], 'fold_idx': [], 'subset_idx': [], 'frame_recall': [], 'doa_error': []
    #     }

    #     num_subsets = len(outputs)

    #     for subset_idx in range(num_subsets):
    #         frame_recall = torch.stack([x['test_frame_recall'] for x in outputs[subset_idx]]).detach().cpu().numpy()
    #         doa_error = torch.stack([x['test_doa_error'] for x in outputs[subset_idx]]).detach().cpu().numpy()

    #         num_sequences = len(frame_recall)

    #         for seq_idx in range(num_sequences):
    #             results['model'].append(self.hparams['name'])
    #             results['dataset'].append(dataset_name)
    #             results['fold_idx'].append(self.cv_fold_idx)
    #             results['subset_idx'].append(subset_idx)
    #             results['frame_recall'].append(frame_recall[seq_idx])
    #             results['doa_error'].append(doa_error[seq_idx])

    #     data_frame = pd.DataFrame.from_dict(results)

    #     if not os.path.isfile(results_file):
    #         data_frame.to_json(results_file)

    #     average_frame_recall = torch.tensor(data_frame['frame_recall'].mean(), dtype=torch.float32)
    #     average_doa_error = torch.tensor(data_frame['doa_error'].mean(), dtype=torch.float32)

    #     self.log_dict({'test_frame_recall': average_frame_recall, 'test_doa_error': average_doa_error})

    def _process_batch(self,
                       batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                       ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        audio_features, targets = batch
        predictions = self.forward(audio_features)

        return predictions, targets

    def train_dataloader(self) -> DataLoader:
        train_dataset = TUTSoundEvents(self.dataset_path, split='train',
                                       tmp_dir=self.hparams['tmp_dir'],
                                       test_fold_idx=self.cv_fold_idx,
                                       sequence_duration=self.hparams['sequence_duration'],
                                       chunk_length=self.hparams['chunk_length'],
                                       frame_length=self.hparams['frame_length'],
                                       num_fft_bins=self.hparams['num_fft_bins'],
                                       max_num_sources=self.hparams['max_num_sources'])

        return DataLoader(train_dataset, shuffle=True, batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'])

    def val_dataloader(self) -> DataLoader:
        valid_dataset = TUTSoundEvents(self.dataset_path, split='valid',
                                       tmp_dir=self.hparams['tmp_dir'],
                                       test_fold_idx=self.cv_fold_idx,
                                       sequence_duration=self.hparams['sequence_duration'],
                                       chunk_length=self.hparams['chunk_length'],
                                       frame_length=self.hparams['frame_length'],
                                       num_fft_bins=self.hparams['num_fft_bins'],
                                       max_num_sources=self.hparams['max_num_sources'])

        return DataLoader(valid_dataset, shuffle=False, batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'])

    def test_dataloader(self) -> List[DataLoader]:
        # During testing, a whole sequence is packed into one batch. The batch size set for training and validation
        # is ignored in this case.
        num_chunks_per_sequence = int(self.hparams['sequence_duration'] / self.hparams['chunk_length'])

        test_loaders = []

        for num_overlapping_sources in range(1, min(self.max_num_overlapping_sources_test,3)):
            test_dataset = TUTSoundEvents(self.dataset_path, split='test',
                                          tmp_dir=self.hparams['tmp_dir'],
                                          test_fold_idx=self.cv_fold_idx,
                                          sequence_duration=self.hparams['sequence_duration'],
                                          chunk_length=self.hparams['chunk_length'],
                                          frame_length=self.hparams['frame_length'],
                                          num_fft_bins=self.hparams['num_fft_bins'],
                                          max_num_sources=self.hparams['max_num_sources'],
                                          num_overlapping_sources=num_overlapping_sources)

            test_loaders.append(DataLoader(test_dataset, shuffle=False, batch_size=num_chunks_per_sequence,
                                           num_workers=self.hparams['num_workers']))

        return test_loaders


###

class FeatureExtraction(nn.Module):
    def __init__(self,
                 chunk_length: float = 2.,
                 frame_length: float = 0.04,
                 num_fft_bins: int = 2048,
                 embedding_dim: int = 32,
                 max_num_sources: int = 3,
                 dropout_rate: float = 0.0) -> None:
        """Feature extraction stage, as described in Sec. 3.1 of the paper. The utilized convolutional neural network
        structure is based on the framework presented in [1].

        Args:
            chunk_length (float): Signal chunk (processing block) length in seconds.
            frame_length (float): Frame length within chunks in seconds.
            num_fft_bins (int): Number of frequency bins used for representing the input spectrograms.
            embedding_dim (int): Dimension of the feature embeddings.
            max_num_sources (int): Maximum number of sources that should be represented by the model.
            dropout_rate (float): Global dropout rate used within this stage.
        """
        super(FeatureExtraction, self).__init__()

        self.conv_network = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True),
            nn.Dropout(p=dropout_rate)
        )

        num_frames_per_chunk = int(2 * chunk_length / frame_length)
        flatten_dim = functools.reduce(operator.mul, list(
            self.conv_network(torch.rand(1, 8, num_frames_per_chunk, num_fft_bins // 2)).shape[slice(1, 4, 2)]))

        self.fc_network = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        self.embedding_output = nn.Linear(128, (embedding_dim - 1) * max_num_sources)
        self.observation_noise_output = nn.Linear(128, embedding_dim * max_num_sources)

        self.embedding_dim = embedding_dim
        self.max_num_sources = max_num_sources

    def forward(self, audio_features: Tensor) -> Tuple[Tensor, Tensor]:
        """Feature extraction forward pass.

        Args:
            audio_features (Tensor): Audio feature tensor (input spectrogram representation).

        Returns:
            tuple: Feature embeddings and estimated (diagonal) observation noise covariance matrices.
        """
        batch_size, _, num_steps, _ = audio_features.shape

        output = self.conv_network(audio_features)
        output = output.permute(0, 2, 1, 3).flatten(2)

        output = self.fc_network(output)

        embeddings = self.embedding_output(output).view(batch_size, num_steps, self.max_num_sources, -1)
        embeddings = embeddings.permute(0, 2, 1, 3)

        positional_encodings = torch.linspace(0, 1, num_steps, device=audio_features.device)[None, ..., None, None]
        positional_encodings = positional_encodings.repeat((batch_size, 1, self.max_num_sources, 1))
        positional_encodings = positional_encodings.permute(0, 2, 1, 3)

        embeddings = torch.cat((embeddings, positional_encodings), dim=-1)

        observation_noise = self.observation_noise_output(output).view(batch_size, num_steps, self.max_num_sources, -1)
        observation_noise = torch.exp(observation_noise.permute(0, 2, 1, 3))

        return embeddings, observation_noise

class LinearGaussianSystem(nn.Module):
    def __init__(self,
                 state_dim: int,
                 observation_dim: int,
                 prior_mean: Tensor = None,
                 prior_covariance: Tensor = None) -> None:
        """Linear Gaussian system, as described in Sec. 3.3 in the paper.

        Args:
            state_dim (int): State dimension.
            observation_dim (int): Observation dimension.
            prior_mean (Tensor): Prior mean vector.
            prior_covariance (Tensor): Prior covariance matrix.
        """
        super(LinearGaussianSystem, self).__init__()

        self.observation_dim = observation_dim

        self.observation_matrix = nn.Parameter(1e-3 * torch.randn((observation_dim, state_dim)), requires_grad=True)
        self.observation_bias = nn.Parameter(1e-6 * torch.randn(observation_dim), requires_grad=True)

        if prior_mean is not None:
            self.prior_mean = nn.Parameter(prior_mean, requires_grad=False)
        else:
            self.prior_mean = nn.Parameter(torch.zeros(state_dim), requires_grad=False)

        if prior_covariance is not None:
            # We use the precision matrix here to avoid matrix inversion during forward pass
            self.prior_precision = nn.Parameter(torch.inverse(prior_covariance), requires_grad=False)
        else:
            self.prior_precision = nn.Parameter(torch.eye(state_dim), requires_grad=False)

    def forward(self,
                observation: Tensor,
                observation_noise: Tensor = None,
                prior_mean: Tensor = None,
                prior_covariance: Tensor = None) -> Tuple[Tensor, Tensor]:
        """Linear Gaussian system forward pass.

        Args:
            observation (Tensor): Observation vector.
            observation_noise (Tensor): Estimated (or fixed) observation noise covariance matrix (per time-step).
            prior_mean (Tensor): Prior mean vector (class-level prior mean vector will not be used).
            prior_covariance (Tensor): Prior covariance matrix (class-level prior covariance matrix will not be used).
        """
        if observation_noise is None:
            observation_noise = 1e-6 * torch.eye(self.observation_dim, device=observation.device)

        observation_noise_precision = torch.inverse(observation_noise)

        if prior_mean is None:
            prior_mean = self.prior_mean

        if prior_covariance is None:
            prior_precision = self.prior_precision
        else:
            prior_precision = torch.inverse(prior_covariance)

        innovation_covariance = self.observation_matrix.t() @ observation_noise_precision @ self.observation_matrix
        posterior_covariance = torch.inverse(prior_precision + innovation_covariance)

        residual = self.observation_matrix.t() @ observation_noise_precision @ (observation - self.observation_bias).unsqueeze(-1)
        posterior_mean = posterior_covariance @ ((prior_precision @ prior_mean).unsqueeze(-1) + residual)

        return posterior_mean, posterior_covariance
