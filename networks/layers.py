import torch


class AFMLayer(nn.Module):
    """Attentonal Factorization Machine models pairwise (order-2) feature
    interactions without linear term and bias.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **attention_factor** : Positive integer, dimensionality of the
         attention network output space.
        - **l2_reg_w** : float between 0 and 1. L2 regularizer strength
         applied to attention network.
        - **dropout_rate** : float between in [0,1). Fraction of the attention net output units to dropout.
        - **seed** : A Python integer to use as random seed.
      References
        - [Attentional Factorization Machines : Learning the Weight of Feature
        Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
    """

    def __init__(
        self,
        in_features,
        attention_factor=4,
        l2_reg_w=0,
        dropout_rate=0,
        seed=1024,
        device="cpu",
    ):
        super(AFMLayer, self).__init__()
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        embedding_size = in_features

        self.attention_W = nn.Parameter(
            torch.Tensor(embedding_size, self.attention_factor)
        )

        self.attention_b = nn.Parameter(torch.Tensor(self.attention_factor))

        self.projection_h = nn.Parameter(torch.Tensor(self.attention_factor, 1))

        self.projection_p = nn.Parameter(torch.Tensor(embedding_size, 1))

        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(
                tensor,
            )

        for tensor in [self.attention_b]:
            nn.init.zeros_(
                tensor,
            )

        self.dropout = nn.Dropout(dropout_rate)

        self.to(device)

    def forward(self, inputs):
        embeds_vec_list = inputs
        row = []
        col = []

        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)

        p = torch.cat(row, dim=1)
        q = torch.cat(col, dim=1)
        inner_product = p * q

        bi_interaction = inner_product
        attention_temp = F.relu(
            torch.tensordot(bi_interaction, self.attention_W, dims=([-1], [0]))
            + self.attention_b
        )

        self.normalized_att_score = F.softmax(
            torch.tensordot(attention_temp, self.projection_h, dims=([-1], [0])), dim=1
        )
        attention_output = torch.sum(self.normalized_att_score * bi_interaction, dim=1)

        attention_output = self.dropout(attention_output)  # training

        afm_out = torch.tensordot(attention_output, self.projection_p, dims=([-1], [0]))
        return afm_out
