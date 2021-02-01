## Unit test only start
import torch
import logging

# import sys
from IPython import embed
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from networks.wrappers import TimeSeriesEncoder
from sklearn.metrics import accuracy_score


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
        pretrain_mat=None,
        **kwargs,
    ):
        super().__init__(architecture="MultiLSTM", **kwargs)
        # super().__init__()

        self.prediction_dims = (
            prediction_dims if prediction_dims else list(range(in_channels))
        )
        self.prediction_length = prediction_length
        self.in_channels = in_channels
        self.vocab_size = vocab_size
        if vocab_size is not None and embedding_dim is not None:
            if pretrain_mat is not None:
                self.embedder = nn.Embedding.from_pretrained(pretrain_mat)
            else:
                self.embedder = nn.Embedding(vocab_size, embedding_dim)
                # nn.init.uniform_(self.embedder.weight, -0.01, 0.01)
            lstm_input_dim = embedding_dim
        else:
            self.embedder = None
            lstm_input_dim = in_channels

        final_output_dim = in_channels * embedding_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.ln = nn.Linear(lstm_input_dim, hidden_size)

        self.max_pooling = nn.MaxPool1d(kernel_size=hidden_size)
        self.linear = nn.Linear(hidden_size, final_output_dim)

        self.dropout = nn.Dropout(dropout)
        # self.loss_fn = nn.MSELoss(reduction="none")
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.compile()

    def forward(self, batch_window):
        # batch_window = batch_window.permute(0, 2, 1)  # b x win x ts_dim
        self.batch_size = batch_window.size(0)
        x, y = (
            batch_window[:, 0 : -self.prediction_length, :],
            batch_window[:, -1, :],
        )

        if self.embedder:
            x_emb = self.embedder(x.long()).mean(-2)  # sum all dims
            y_emb = self.embedder(y.long())
            y_neg = torch.randint(
                0, self.vocab_size, (self.batch_size, self.in_channels * 5)
            ).to(self.device)
            y_neg_emb = self.embedder(y_neg.long())

        # print(self.embedder.weight[0])

        # lstm_out, lstm_hidden = self.lstm(x_emb)
        # outputs = lstm_out.mean(dim=1)  # b x
        # outputs = self.dropout(outputs)
        # embed()
        outputs = self.ln(x_emb.mean(dim=1))

        outputs = self.linear(outputs).reshape(self.batch_size, self.in_channels, -1)

        y_emb = y_emb.reshape(self.batch_size, self.in_channels, -1)

        # embed()
        loss_pos = self.loss_fn(
            torch.bmm(outputs, y_emb.detach().transpose(-1, 1)).mean(dim=1),
            torch.ones((self.batch_size, 1)),
        )

        loss_neg = self.loss_fn(
            torch.bmm(outputs, y_neg_emb.detach().transpose(-1, 1)).mean(dim=(1, 2)),
            torch.zeros((self.batch_size)),
        )

        # embed()
        # loss_pos = F.logsigmoid(torch.bmm(outputs, y_emb.transpose(-1, 1)).mean())

        # # embed()
        # loss_neg = F.logsigmoid(-torch.bmm(outputs, y_neg_emb.transpose(-1, 1)).mean())

        loss = loss_pos

        logging.info(
            "ploss: {:.4f}, nloss: {:.4f}, pos: {:.4f}, neg: {:.4f}".format(
                loss_pos.item(),
                loss_neg.item(),
                # 0,
                torch.bmm(outputs, y_emb.transpose(-1, 1)).mean(),
                torch.bmm(outputs, y_neg_emb.transpose(-1, 1)).mean(),
            )
        )

        queries = outputs.reshape(self.batch_size * self.in_channels, -1)
        # print(queries.sum(dim=-1))
        base = self.embedder.weight

        score_mat = torch.matmul(queries, base.T)
        # embed()
        # for k in score_mat.detach().cpu().numpy():
        #     plt.plot(k)
        # for k in y.detach().cpu().numpy():
        #     plt.plot(k, 1, "ro")
        # plt.show()
        max_score, recst = score_mat.max(dim=-1)

        # embed()
        # print(queries)
        # print(max_score[0].item(), score_mat[0, y.long().reshape(-1)[0]].item())

        recst = recst.reshape(self.batch_size, self.in_channels)

        logging.info(
            "acc: {}".format(
                accuracy_score(
                    recst.reshape(-1).cpu().numpy(), y.reshape(-1).cpu().numpy()
                )
            )
        )

        score = (outputs * y_emb).sum(dim=-1)

        return_dict = {
            "loss": loss,
            "recst": recst,
            "repr": outputs,
            "score": score,
            "y": y,
        }

        return return_dict


if __name__ == "__main__":
    inp = torch.randn((32, 1, 46))
    model = MultiLSTMEncoder(in_channels=25, num_layers=1, window_size=45)
    out = model.forward(inp)
