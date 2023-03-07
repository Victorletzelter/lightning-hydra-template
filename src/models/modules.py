import abc
from src.data.data_handlers import TUTSoundEvents
import math
from src.metrics import frame_recall, doa_error
import numpy as np
import os
import pandas as pd
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.optim as optim
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
            
            results_file = os.path.join(self.hparams['results_dir'],
                                        'max_sources'+ str(self.max_num_sources) +  '_' +
                                        'num_test_samples'+ str(len(data_frame['frame_recall'])) + '_'
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
                                    'num_test_samples'+ str(len(data_frame['frame_recall'])) + '_'
                                    + '.json')
            
            if not os.path.isfile(results_file):
                with open(str(results_file),'w') as file : 
                    json.dump(_results, file)

        self.log_dict({'test_frame_recall': average_frame_recall, 'test_doa_error': average_doa_error})

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

        for num_overlapping_sources in range(1, min(self.max_num_overlapping_sources_test,3)+1):
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


class FeatureExtraction(nn.Module):
    """CNN-based feature extraction originally proposed in [1].

    Args:
        num_steps_per_chunk: Number of time steps per chunk, which is required for correct layer normalization.

        num_fft_bins: Number of FFT bins used for spectrogram computation.

        dropout_rate: Dropout rate.

    References:
        [1] Sharath Adavanne, Archontis Politis, Joonas Nikunen, and Tuomas Virtanen, "Sound event localization and
            detection of overlapping sources using convolutional recurrent neural network" in IEEE Journal of Selected
            Topics in Signal Processing (JSTSP 2018)
    """
    def __init__(self,
                 num_steps_per_chunk: int,
                 num_fft_bins: int,
                 dropout_rate: float = 0.0) -> None:
        super(FeatureExtraction, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True),
            nn.Dropout(p=dropout_rate)
        )

        self.layer_norm = nn.LayerNorm([num_steps_per_chunk, int(num_fft_bins / 4)])

    def forward(self,
                audio_features: torch.Tensor) -> torch.Tensor:
        """Feature extraction forward pass.

        :param audio_features: Input tensor with dimensions [NxTxBx2*C], where N is the batch size, T is the number of
                               time steps per chunk, B is the number of FFT bins and C is the number of audio channels.
        :return: Extracted features with dimension [NxTxB/4].
        """
        output = self.conv_layer1(audio_features)     
        output = self.conv_layer2(output)    
        output = self.conv_layer3(output)
        output = output.permute(0, 2, 1, 3)
        batch_size, num_frames, _, _ = output.shape
        output = output.contiguous().view(batch_size, num_frames, -1)

        return self.layer_norm(output)


class LocalizationOutput(nn.Module):
    """Implements a module that outputs source activity and direction-of-arrival for sound event localization. An input
    of fixed dimension is passed through a fully-connected layer and then split into a source activity vector with
    sigmoid output activations and corresponding azimuth and elevation vectors, which are subsequently combined to a
    direction-of-arrival output tensor.

    Args:
        input_dim: Input dimension.

        max_num_sources: Maximum number of sound sources that should be represented by the module.
    """
    def __init__(self, input_dim: int, max_num_sources: int):
        super(LocalizationOutput, self).__init__()

        self.source_activity_output = nn.Linear(input_dim, max_num_sources)
        self.azimuth_output = nn.Linear(input_dim, max_num_sources)
        self.elevation_output = nn.Linear(input_dim, max_num_sources)

    def forward(self,
                input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward pass.

        :param input: Input tensor with dimensions [NxTxD], where N is the batch size, T is the number of time steps per
                      chunk and D is the input dimension.
        :return: Tuple containing the source activity tensor of size [NxTxS] and the direction-of-arrival tensor with
                 dimensions [NxTxSx2], where S is the maximum number of sources.
        """
        source_activity = self.source_activity_output(input)

        azimuth = self.azimuth_output(input)
        elevation = self.elevation_output(input)
        direction_of_arrival = torch.cat((azimuth.unsqueeze(-1), elevation.unsqueeze(-1)), dim=-1)

        return source_activity, direction_of_arrival

### Time distributed fully connected layers
class TimeDistributedFC(nn.Module):
    """
    Class for applying time distributed fully connected layers with input tensors 
    of shape  [N, T, in_features] where N is the batch_size, T is the number of time step,
    and in_features is the number of input features at each time step. 
    Output tensors will have shape [N, T, out_features], where out_features is the number 
    of output features at each time step.
    """
    def __init__(self, in_features, out_features):
        super(TimeDistributedFC, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        
        N,T = x.shape[0],x.shape[1]
        
        # reshape input tensor to [N*T x B/4]
        x = x.view(-1, x.shape[-1])

        # apply linear layer to each time step
        x = self.linear(x)

        # reshape output tensor to [N x T x S]
        x = x.view(N, T, -1)

        return x       

### MH for multihypothesis.  
class MHLocalizationOutput(nn.Module):
    """Implements a module that outputs source activity and direction-of-arrival for sound event localization. An input
    of fixed dimension is passed through a fully-connected layer and then split into a source activity vector with
    sigmoid output activations and corresponding azimuth and elevation vectors, which are subsequently combined to a
    direction-of-arrival output tensor.
    Args:
        input_dim: Input dimension.
        
        num_hypothesis: Number of hypothesis in the model. (TODOO)
    """
    def __init__(self, input_dim: int, num_hypthesis: int, output_dim: int=2):
        super(LocalizationOutput, self).__init__()

        self.source_activity_output_layers = {}
        self.azimuth_output_layers = {}
        self.elevation_output_layers = {}
        self.doa_layers = {}
        self.num_hypothesis = num_hypthesis
        self.output_dim = output_dim
        
        for k in range(self.num_hypothesis) :  
            # self.azimuth_output_layers['hyp_'+'{}'.format(i)] = TimeDistributedFC(input_dim, self.num_hypothesis)
            # self.elevation_output_layers['hyp_'+'{}'.format(i)] = TimeDistributedFC(input_dim, self.num_hypothesis)
            self.doa_layers['hyp_'+'{}'.format(k)] = TimeDistributedFC(input_dim, output_dim)
            
        ### OR 
        # self.doa_layers = TimeDistributedFC(input_dim, self.num_hypothesis*2) #if output_dim=2

    def forward(self,
                input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward pass.

        :param input: Input tensor with dimensions [NxTxD], where N is the batch size, T is the number of time steps per
                    chunk and D is the input dimension.
        :return: Stacked direciton of arrival hypothesis with shape [NxTxself.num_hypothesisxoutput_dim]
        """  
        directions_of_arrival = []
        
        for k in range(self.num_hypothesis) :
            directions_of_arrival.append(self.doa_layers['hyp_'+'{}'.format(k)](input)) # Size [NxTx2]
            
        hyp_stacked = torch.stack(directions_of_arrival, dim=-2) #Shape [NxTxself.num_hypothesisx2]

        ### OR 
        # directions_of_arrival = self.doa_layers(input) # Size [NxTx(2*self.num_hypothesis)]
        # hyps_splitted = torch.split(directions_of_arrival, [2 for i in range(num_hypothesis)], 2) #num_hypothesis-uples of elements of shape [N,T,2]
        # hyps_stacked = torch.stack([h for h in hyps_splitted], dim=2) #Tuples of elements of shape [N,T,num_hypothesis,2]

        return hyp_stacked