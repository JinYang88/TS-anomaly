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
        **kwargs
    ):
        super().__init__(architecture="MultiLSTM", **kwargs)
        # super().__init__()

        if vocab_size is not None and embedding_dim is not None:
            self.embedder = nn.Embedding(vocab_size, embedding_dim)
            lstm_input_dim = embedding_dim
        else:
            self.embedder = None
            lstm_input_dim = in_channels

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.max_pooling = nn.MaxPool1d(kernel_size=hidden_size)
        # self.linear = nn.Linear(hidden_size, in_channels)
        self.linear = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, in_channels))

        self.loss_fn = nn.MSELoss(reduction="none")

        self.compile()

    def forward(self, batch_window):
        # batch_window = batch_window.permute(0, 2, 1)  # b x win x ts_dim
        batch_window, y = batch_window[:, 0:-1, :], batch_window[:, -1, :]

        if self.embedder:
            batch_window = self.embedder(batch_window.long().squeeze())

        lstm_out, lstm_hidden = self.lstm(batch_window)
        outputs = lstm_out.sum(dim=1)  # consider every dim of all timestamps

        recst = self.linear(outputs)
        loss = self.loss_fn(recst, y)
        return_dict = {
            "loss": loss.sum(),
            "recst": recst,
            "repr": outputs,
            "diff": loss,
            "anomaly_label": y,
        }
        return return_dict


if __name__ == "__main__":
    inp = torch.randn((32, 1, 46))
    model = MultiLSTMEncoder(in_channels=25, num_layers=1, window_size=45)
    out = model.forward(inp)
