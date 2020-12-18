## Unit test only start
# import torch
# import sys
from IPython import embed
from torch import nn

from networks.wrappers import TimeSeriesEncoder

# sys.path.append("../")
## Unit test only end


class MultiLSTMEncoder(TimeSeriesEncoder):
    """
    Encoder of a time series using a LSTM, ccomputing a linear transformation
    of the output of an LSTM

    Takes as input a three-dimensional tensor (`B`, `L`, `C`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a two-dimensional tensor (`B`, `C`).
    """

    def __init__(
        self,
        in_channels,
        hidden_size=64,
        num_layers=1,
        vocab_size=None,
        embedding_dim=None,
        dropout=0,
        prediction_length=1,
        prediction_dims=[],
        **kwargs,
    ):
        super().__init__(architecture="MultiLSTM", **kwargs)
        # super().__init__()

        self.prediction_dims = (
            prediction_dims if prediction_dims else list(range(in_channels))
        )
        self.prediction_length = prediction_length

        if vocab_size is not None and embedding_dim is not None:
            self.embedder = nn.Embedding(vocab_size, embedding_dim)
            lstm_input_dim = embedding_dim
        else:
            self.embedder = None
            lstm_input_dim = in_channels

        final_output_dim = prediction_length * len(self.prediction_dims)

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.max_pooling = nn.MaxPool1d(kernel_size=hidden_size)
        self.linear = nn.Linear(hidden_size, final_output_dim)

        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.MSELoss(reduction="none")

        self.compile()

    def forward(self, batch_window):
        # batch_window = batch_window.permute(0, 2, 1)  # b x win x ts_dim
        self.batch_size = batch_window.size(0)
        x, y = (
            batch_window[:, 0 : -self.prediction_length, :],
            batch_window[:, -self.prediction_length :, self.prediction_dims],
        )

        embed()

        if self.embedder:
            x = self.embedder(x.long().squeeze())

        lstm_out, lstm_hidden = self.lstm(x)
        outputs = lstm_out.sum(dim=1)  # b x
        # outputs = lstm_out[:, -1, :]  # b x
        outputs = self.dropout(outputs)

        recst = self.linear(outputs).view(
            self.batch_size, self.prediction_length, len(self.prediction_dims)
        )
        loss = self.loss_fn(recst, y)
        return_dict = {
            "loss": loss.mean(),
            "recst": recst,
            "repr": outputs,
            "diff": loss,
            "y": y,
        }

        return return_dict


if __name__ == "__main__":
    inp = torch.randn((32, 1, 46))
    model = MultiLSTMEncoder(in_channels=25, num_layers=1, window_size=45)
    out = model.forward(inp)
