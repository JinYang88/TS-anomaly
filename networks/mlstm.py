## Unit test only start
# import torch
# import sys
import torch
from IPython import embed
from torch import nn
from networks.wrappers import TimeSeriesEncoder


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
        inter="FM",
        **kwargs,
    ):
        super().__init__(architecture="MultiLSTM", **kwargs)

        self.prediction_dims = (
            prediction_dims if prediction_dims else list(range(in_channels))
        )
        self.prediction_length = prediction_length
        self.inter = inter

        if vocab_size is not None and embedding_dim is not None:
            self.embedder = nn.Embedding(vocab_size, embedding_dim)
            lstm_input_dim = embedding_dim
        else:
            self.embedder = None
            lstm_input_dim = in_channels

        final_output_dim = prediction_length * len(self.prediction_dims)

        if self.inter == "TIME" or self.inter == "MEAN":
            clf_input_dim = in_channels
        elif self.inter == "DIM":
            clf_input_dim = kwargs["window_size"] - 1
        elif self.inter == "CONCAT":
            clf_input_dim = in_channels * (kwargs["window_size"] - 1)
        elif self.inter == "FM":
            clf_input_dim = 2 * (kwargs["window_size"] - 1 + in_channels)
        else:
            clf_input_dim = kwargs["window_size"] - 1 + in_channels

        self.res_w = nn.Linear(
            in_channels * (kwargs["window_size"] - 1),
            kwargs["window_size"] - 1 + in_channels,
        )

        self.linear = nn.Sequential(
            nn.Linear(clf_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, final_output_dim),
        )

        # self.linear = nn.Sequential(nn.Linear(clf_input_dim, final_output_dim))

        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.MSELoss(reduction="none")

        self.compile()

    def FM_interaction(self, x):
        """
        shape of x: (batchsize, dims, feature lens)
        """
        sum_of_square = torch.sum(x, dim=1) ** 2
        square_of_sum = torch.sum(x ** 2, dim=1)
        bi_interaction_vector = (sum_of_square - square_of_sum) * 0.5
        # return bi_interaction_vector.sum(dim=-1, keepdim=True)
        return bi_interaction_vector

    def forward(self, batch_window):
        # batch_window = batch_window.permute(0, 2, 1)  # b x win x ts_dim
        self.batch_size = batch_window.size(0)
        x, y = (
            batch_window[:, 0 : -self.prediction_length, :],
            batch_window[:, -self.prediction_length :, self.prediction_dims],
        )

        if self.inter == "FM":
            time_inter = self.FM_interaction(x)
            dim_inter = self.FM_interaction(x.transpose(2, 1))
            # print(dim_inter.shape)
            raw = self.res_w(x.reshape(self.batch_size, -1))
            inter = torch.cat([time_inter, dim_inter], dim=-1)
            outputs = torch.cat([raw, inter])

        elif self.inter == "MEAN":
            outputs = x.mean(dim=1)
        elif self.inter == "TIME":
            time_inter = self.FM_interaction(x)
            outputs = time_inter
        elif self.inter == "DIM":
            dim_inter = self.FM_interaction(x.transpose(2, 1))
            outputs = dim_inter
        elif self.inter == "CONCAT":
            outputs = x.reshape(self.batch_size, -1)

        outputs = self.dropout(outputs)
        recst = self.linear(outputs).view(
            self.batch_size, self.prediction_length, len(self.prediction_dims)
        )
        loss = self.loss_fn(recst, y)
        return_dict = {
            "loss": loss.sum(),
            "recst": recst,
            "repr": outputs,
            "score": loss,
            "y": y,
        }

        return return_dict


if __name__ == "__main__":
    inp = torch.randn((32, 1, 46))
    model = MultiLSTMEncoder(in_channels=25, num_layers=1, window_size=45)
    out = model.forward(inp)
