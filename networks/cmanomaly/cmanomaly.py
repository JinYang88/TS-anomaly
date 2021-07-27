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

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.prediction_dims = (
            prediction_dims if prediction_dims else list(range(in_channels))
        )
        self.prediction_length = prediction_length
        self.gamma = gamma

        self.embedder = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True)

        final_output_dim = 26
        self.predcitor = nn.Sequential(
            # nn.Linear(embedding_dim * in_channels * (window_size-1), 128),
            # nn.Linear(128 , 128),
            nn.Linear(embedding_dim * (window_size-1) , 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels * final_output_dim),
        )
        self.dropout = nn.Dropout(dropout)
        # self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
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
        # batch_window = batch_window.permute(0, 2, 1)  # b x win x ts_dim
        x, y = input_dict["x"].to(self.device), input_dict["y"].to(self.device)
        self.batch_size = x.size(0)

        x_embed = self.embedder(x.long()).view(-1, self.in_channels, self.embedding_dim)
        # interaction, interaction_score = self.afm(x_embed)
        interaction = self.CM_interaction(x_embed)
        # # interaction = x_embed.mean(dim=1)

        representation = interaction.view(self.batch_size, -1, self.embedding_dim)
        # # print(x.shape, representation.shape, representation[0])
        # lstm_out, _ = self.lstm(representation)
        # lstm_out = self.dropout(lstm_out[:, -1, :])

        # lstm_out = x_embed.view(self.batch_size, -1) # only this -> f1 score 0.78!
        lstm_out = representation.view(self.batch_size, -1)
        recst = self.predcitor(lstm_out).view(-1, 26) # batch*channel x 26
        y = y.view(-1)
        loss = self.loss_fn(recst, y)

        return_dict = {
            "loss": loss.mean(),
            "recst": recst.argmax(dim=-1).view(self.batch_size, -1),
            "score": loss.view(self.batch_size, -1),
            "y": y.view(self.batch_size, -1),
        }

        return return_dict
