import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.utils import scatter
from torch.nn import Linear

from model.input_features import lig_feature_dims, edge_feature_dims
from model.model_utils import MLP, AtomEncoder


GNN_LAYER = dict(GAT=GATConv, GCN=GCNConv, Transformer=TransformerConv)


class GNN(torch.nn.Module):
    def __init__(self, output_features, model_config):
        super().__init__()
        self.use_edge_attribution = model_config.use_edge_attribution

        if model_config.atom_embedder:
            self.embed_nodes = AtomEncoder(
                emb_dim=model_config.hidden_channels,
                feature_dims=lig_feature_dims,
                sigma_embed_dim=0,
            )
        else:
            self.embed_nodes = MLP(
                channel_list=[lig_feature_dims[0]]
                + [model_config.hidden_channels] * model_config.layer_embed_nodes
            )

        if self.use_edge_attribution:
            self.embed_edges = MLP(
                channel_list=[edge_feature_dims]
                + [model_config.hidden_channels] * model_config.layer_embed_edges
            )

        self.layers = torch.nn.ModuleList()
        for i in range(model_config.nb_layer):
            conv = GNN_LAYER[model_config.name](
                model_config.hidden_channels, model_config.hidden_channels
            )
            self.layers.append(conv)

        self.output_layer = Linear(model_config.hidden_channels, output_features)
        self.model_config = model_config

    def forward(self, x, edge_index, edge_attr, batch_idx):
        x = self.embed_nodes(x)
        edge_attr = self.embed_edges(edge_attr) if self.use_edge_attribution else None

        for i in range(self.model_config.nb_layer):
            x = self.layers[i](x, edge_index, edge_attr=edge_attr)
            x = x.relu()
            x = F.dropout(x, p=self.model_config.dropout, training=self.training)

        x = scatter(x, batch_idx, dim=0, reduce="mean")

        # output
        x = self.output_layer(x)

        return x
