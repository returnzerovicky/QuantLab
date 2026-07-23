"""
QuantLab AI
LSTM, GRU and Transformer Models
"""

import torch
import torch.nn as nn

from .config import config


class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(
    input_size=config.input_size,
    hidden_size=config.hidden_size,
    num_layers=config.num_layers,
    batch_first=True,
    dropout=config.dropout if config.num_layers > 1 else 0.0
)

        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(
            config.hidden_size,
            config.output_size
        )

    def forward(self, x):

        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device
        )

        c0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device
        )

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        out = self.dropout(out)

        out = self.fc(out)

        return out


class GRUModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.gru = nn.GRU(
    input_size=config.input_size,
    hidden_size=config.hidden_size,
    num_layers=config.num_layers,
    batch_first=True,
    dropout=config.dropout if config.num_layers > 1 else 0.0
)

        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(
            config.hidden_size,
            config.output_size
        )

    def forward(self, x):

        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device
        )

        out, _ = self.gru(x, h0)

        out = out[:, -1, :]

        out = self.dropout(out)

        out = self.fc(out)

        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Linear(
            config.input_size,
            config.hidden_size
        )

        self.position = PositionalEncoding(
            config.hidden_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8,
            batch_first=True,
            dropout=config.dropout
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(
            config.hidden_size,
            config.output_size
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.position(x)

        x = self.encoder(x)

        x = x[:, -1, :]

        x = self.dropout(x)

        x = self.fc(x)

        return x


def build_model(model_name="lstm"):

    model_name = model_name.lower()

    if model_name == "lstm":
        return LSTMModel()

    if model_name == "gru":
        return GRUModel()

    if model_name == "transformer":
        return TransformerModel()

    raise ValueError(
        f"Unknown model: {model_name}"
    )
