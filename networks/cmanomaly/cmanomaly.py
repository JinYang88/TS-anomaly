## Unit test only start
# import torch
# import sys
import torch
import itertools
import torch.nn.functional as F
from IPython import embed
from torch import nn
from networks.wrappers import TimeSeriesEncoder


class AFMLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_fields, attn_size=32, dropout=0.1):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.row, self.col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        p, q = x[:, self.row], x[:, self.col]
        inner_product = p * q
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        attn_scores = self.dropout(attn_scores)
        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        attn_output = self.dropout(attn_output)
        return attn_output, attn_scores


class CMAnomaly(TimeSeriesEncoder):
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
        window_size,
        vocab_size=None,
        embedding_dim=None,
        dropout=0,
        prediction_length=1,
        prediction_dims=[],
        gamma=0.01,
        **kwargs,
    ):
        super().__init__(architecture="CMAomaly", **kwargs)

        self.prediction_dims = (
            prediction_dims if prediction_dims else list(range(in_channels))
        )
        self.prediction_length = prediction_length
        self.gamma = gamma

        final_output_dim = prediction_length * len(self.prediction_dims)

        self.embed = nn.Linear(in_channels, embedding_dim)
        clf_input_dim = embedding_dim + window_size - 1
        self.time_afm = AFMLayer(embedding_dim, num_fields=window_size - 1)
        self.feat_afm = AFMLayer(window_size - 1, num_fields=embedding_dim)

        self.res_w = nn.Linear(
            embedding_dim * (window_size - 1),
            window_size - 1 + embedding_dim,
        )

        self.linear = nn.Sequential(
            nn.Linear(clf_input_dim, final_output_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(64, final_output_dim),
        )

        # self.linear = nn.Sequential(nn.Linear(clf_input_dim, final_output_dim))

        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.MSELoss(reduction="none")

        self.compile()

    def CM_interaction(self, x):
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

        x = self.embed(x)

        time_inter, time_attn_scores = self.time_afm(x)
        dim_inter, dim_attn_scores = self.feat_afm(x.transpose(2, 1))

        interactions = torch.cat([time_inter, dim_inter], dim=-1)
        outputs = self.dropout(interactions)
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
            "time_attn_scores": time_attn_scores,
            "dim_attn_scores": dim_attn_scores,
        }

        return return_dict


if __name__ == "__main__":
    inp = torch.randn((32, 1, 46))
    model = MultiLSTM(in_channels=25, num_layers=1, window_size=45)
    out = model.forward(inp)
