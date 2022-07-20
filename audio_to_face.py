import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from einops import repeat, rearrange, reduce


class AudioToFace(nn.Module):
    def __init__(
        self,
        audio_size,
        lstm_input_size,
        act_layer,
        hidden_size,
        num_layers=1,
        bias=False,
        batch_first=False,
        dropout=0,
    ):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(118, lstm_input_size),
        )
        self.bn = nn.BatchNorm1d(lstm_input_size)
        self.lstm_layers = nn.LSTM(
            lstm_input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional=True,
        )
        self.lstm_num_layers = num_layers
        self.lstm_hidden_size = hidden_size

    def forward(
        self, 
        audio,
        init,
        lengths,
    ):
        audio = torch.cat((audio, init.repeat(1, audio.shape[1], 1)), 2)
        audio = self.fusion(audio)
        n, T, C = audio.shape
        audio = self.bn(audio.reshape(-1, 192)).reshape(n, T, C)
        lengths = lengths.cpu().tolist()
        audio = rnn_utils.pack_padded_sequence(audio, lengths, batch_first=True, enforce_sorted=False)

        audio, _ = self.lstm_layers(audio)

        audio, length_unpacked = rnn_utils.pad_packed_sequence(audio, batch_first=True)

        return audio
