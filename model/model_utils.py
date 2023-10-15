import torch
from torch.nn import Linear


class MLP(torch.nn.Module):
    def __init__(self, channel_list):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                Linear(channel_list[i - 1], channel_list[i])
                for i in range(1, len(channel_list))
            ]
        )

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            x = x.relu()

        return x


# Adapted from https://github.com/gcorso/DiffDock/blob/main/models/score_model.py#L119
class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims, sigma_embed_dim=0):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(
                x[
                    :,
                    self.num_categorical_features : self.num_categorical_features
                    + self.num_scalar_features,
                ]
            )
        return x_embedding
