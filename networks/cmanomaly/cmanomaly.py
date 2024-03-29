## Unit test only start
# import torch
# import sys
from common.utils import print_to_json
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
        nb_classes,
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

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.prediction_dims = (
            prediction_dims if prediction_dims else list(range(in_channels))
        )
        self.prediction_length = prediction_length
        self.gamma = gamma
        self.nb_classes = nb_classes

        self.embedder = nn.Embedding(vocab_size, embedding_dim)
        self.predcitor = nn.Sequential(
            # nn.Linear(embedding_dim * in_channels * (window_size-1), 128),
            # nn.Linear(128 , 128),
            nn.Linear(2 * embedding_dim * (window_size - 1), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, nb_classes),
        )
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        # self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        # self.loss_fn = nn.MSELoss(reduction="none")
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

    def forward(self, input_dict):
        # x: b x window_size x in_channels x embedding_dim
        x, y = input_dict["x"].to(self.device), input_dict["y"].to(self.device)
        self.batch_size = x.size(0)

        x_embed = self.embedder(x.long()).view(-1, self.in_channels, self.embedding_dim)
        interaction = self.CM_interaction(x_embed)
        repre_self = x_embed.mean(dim=1).view(self.batch_size, -1, self.embedding_dim)
        repre_inter = interaction.view(
            self.batch_size, -1, self.embedding_dim
        )  # b x window x embedding
        representation = torch.cat([repre_self, repre_inter], dim=-1)

        lstm_out = representation.flatten(start_dim=1)
        recst = self.predcitor(lstm_out)  # batch*channel x 26

        loss = self.loss_fn(recst, y)
        return_dict = {
            "loss": loss.mean(),
            "recst": recst,
            "score": loss,
            "y": y,
            "x": x,
        }

        return return_dict
