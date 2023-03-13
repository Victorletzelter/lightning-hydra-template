import argparse
from .modules import AbstractLocalizationModule, FeatureExtraction, LocalizationOutput
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from src.utils import utils
from src.utils.utils import SELLoss

class ADRENALINEEncoder(nn.Module):
    """This class implements the encoder module for a sequence-to-sequence-based sound event localization neural
    network. It uses a feature extraction front-end based on convolutional layers, as proposed in

        Sharath Adavanne, Archontis Politis, Joonas Nikunen, Tuomas Virtanen: "Sound Event Localization and Detection
            of Overlapping Sources Using Convolutional Recurrent Neural Networks" (2018)

    and implements a standard encoder structure based on gated recurrent units.
    """
    def __init__(self,
                 hparams: argparse.Namespace) -> None:
        super(ADRENALINEEncoder, self).__init__()

        self.hidden_dim = hparams.hidden_dim
        self.num_layers = hparams.num_layers

        num_steps_per_chunk = int(2 * hparams.chunk_length / hparams.frame_length)
        self.feature_extraction = FeatureExtraction(num_steps_per_chunk,
                                                    hparams.num_fft_bins,
                                                    dropout_rate=hparams.dropout_rate) #CNN feature extractor. 

        feature_dim = int(hparams.num_fft_bins / 4) #Number of features at each time step after passing through the features extractor.

        self.initial_state = nn.Parameter(
            torch.randn((2 * hparams.num_layers, 1, hparams.hidden_dim), dtype=torch.float32), requires_grad=True
        )

        self.gru = nn.GRU(feature_dim, hparams.hidden_dim, batch_first=True, bidirectional=True,
                          num_layers=hparams.num_layers, dropout=hparams.dropout_rate) # Bidirectional GRU

    def forward(self,
                audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        extracted_features = self.feature_extraction(audio_features) # Shape (batch, num_steps, feature_dim)
        batch_size = extracted_features.shape[0]

        output, hidden = self.gru(extracted_features, self.initial_state.repeat(1, batch_size, 1))
        # output of shape (batch, num_steps, 2*hidden_dim)

        hidden = hidden.view(2, 2, batch_size, -1).permute(0, 2, 3, 1).reshape(self.num_layers, batch_size, 2 * self.hidden_dim)
        # hidden of shape (num_layers, batch_size, 2*hidden_dim)

        return output, hidden


class ADRENALINEDecoder(nn.Module):
    """This class implements an attention-based decoder module for a sequence-to-sequence-based sound event localization
    neural network. It exploits a standard architecture based on the scaled dot-product for computing attention values
    and gated recurrent units as the recurrent part.
    """
    def __init__(self,
                 hparams: argparse.Namespace) -> None:
        super(ADRENALINEDecoder, self).__init__()

        self.hidden_dim = hparams.hidden_dim
        self.num_layers = hparams.num_layers

        self.scale_matrix = nn.Linear(2 * hparams.hidden_dim, 2 * hparams.hidden_dim, bias=False)

        self.gru = nn.GRU(2 * hparams.hidden_dim + 3 * hparams.max_num_sources, 2 * hparams.hidden_dim,
                          batch_first=True, num_layers=hparams.num_layers, dropout=hparams.dropout_rate)

        self.localization_output = LocalizationOutput(2 * hparams.hidden_dim, max_num_sources=hparams.max_num_sources)

    def forward(self,
                source_activity_input: torch.Tensor,
                direction_of_arrival_input: torch.Tensor,
                hidden: torch.Tensor,
                encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param source_activity_input: input vector indicating source activity at the current time step (batch_size, 1)
        :param direction_of_arrival_input: direction-of-arrival input vector at the current time step (batch_size, 2)
        :param hidden: decoder hidden state from the previous time step
        :param encoder_outputs: all encoder outputs (batch_size, num_steps, 2 * hidden_dim)
        :return: source_activity_output, direction_of_arrival_output, hidden: corresponding outputs
        """
        batch_size, sequence_length, _ = encoder_outputs.shape 

        # Compute attention weights via dot product between current hidden state and encoder outputs.
        expanded_hidden = hidden[self.num_layers - 1, :].unsqueeze(0).permute(1, 0, 2).repeat(1, sequence_length, 1) # (batch_size, num_steps, 2 * hidden_dim)

        scaled_dot_product = (self.scale_matrix(encoder_outputs) * expanded_hidden).sum(-1) / np.sqrt(self.hidden_dim) # (batch_size, num_steps) ???
        attention_weights = torch.softmax(scaled_dot_product, dim=-1) # (batch_size, num_steps) ???

        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs) # (batch_size, 1, 2 * hidden_dim)

        input_with_context = torch.cat((source_activity_input,
                                        direction_of_arrival_input.view(batch_size, 1, -1),
                                        context_vector), dim=-1) # (batch_size, 1, 2 * hidden_dim + 3 * max_num_sources)

        output, next_hidden = self.gru(input_with_context, hidden) 
        # output of shape (batch_size, 1, 2 * hidden_dim)
        # next_hidden (num_layers, batch_size, 2 * hidden_dim)

        source_activity_output, direction_of_arrival_output = self.localization_output(output) 
        # source_activity_output, shape (batch_size, 1, max_num_sources)
        # direction_of_arrival_output, (batch_size, 1, 2 * max_num_sources)

        return source_activity_output, direction_of_arrival_output, next_hidden, attention_weights
        # Next hidden of shape (num_layers, batch_size, 2 * hidden_dim)
        # Attention weights of shape 


class ADRENALINE(AbstractLocalizationModule):
    """Implementation of the Attention-based Deep REcurrent Network for locALizINg acoustic Events (ADRENALINE)."""
    def __init__(self,
                 dataset_path: str,
                 cv_fold_idx: int,
                 hparams: argparse.Namespace) -> None:
        super(ADRENALINE, self).__init__(dataset_path, cv_fold_idx, hparams)

        self.max_num_sources = hparams.max_num_sources

        self.encoder = ADRENALINEEncoder(hparams)
        self.decoder = ADRENALINEDecoder(hparams)

    def get_loss_function(self) -> nn.Module:
        return SELLoss(self.hparams.max_num_sources, alpha=self.hparams.alpha)

    def forward(self,
                audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        batch_size, _, sequence_length, _ = audio_features.shape
        device = audio_features.device

        source_activity = torch.zeros((batch_size, 1, self.max_num_sources)).to(device) # (batch_size, 1, max_num_sources)
        direction_of_arrival = torch.zeros((batch_size, 1, self.max_num_sources, 2)).to(device)  # (batch_size, 1, max_num_sources,2)

        encoder_outputs, hidden = self.encoder(audio_features)
        # encoder_outputs of shape (batch, num_steps, 2*hidden_dim)
        # hidden of shape (num_layers, batch_size, 2*hidden_dim)
        source_activity_output = torch.zeros((batch_size, sequence_length, self.max_num_sources)).to(device) # (batch_size, sequence_length, max_num_sources)
        direction_of_arrival_output = torch.zeros((batch_size, sequence_length, self.max_num_sources, 2)).to(device) # (batch_size, sequence_length, max_num_sources, 2)

        attention_map = []

        for step_idx in range(sequence_length):
            source_activity, direction_of_arrival, hidden, attention_weights = self.decoder(
                source_activity, direction_of_arrival, hidden, encoder_outputs)
            # source_activity_output, shape (batch_size, 1, max_num_sources)
            # direction_of_arrival, (batch_size, 1, 2 * max_num_sources)
            # hidden 
            # attention_weights 
            attention_map.append(attention_weights.unsqueeze(-1))
            
            if self.max_num_sources == 1 : 
                source_activity_output[:, step_idx, :] = source_activity.squeeze().unsqueeze(-1)
                direction_of_arrival_output[:, step_idx, :, :] = direction_of_arrival.squeeze().unsqueeze(-2)
                
            else : 
                source_activity_output[:, step_idx, :] = source_activity.squeeze()
                direction_of_arrival_output[:, step_idx, :, :] = direction_of_arrival.squeeze()

        attention_map = torch.cat(attention_map, dim=-1)

        meta_data = {
            'attention_map': attention_map
        }

        return source_activity_output, direction_of_arrival_output, meta_data
