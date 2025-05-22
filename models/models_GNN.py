import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) model for generating node embeddings.

    Args:
        in_channels (int): Dimension of input node features.
        hidden_channels (int): Dimension of hidden layer.
        out_channels (int): Dimension of output embeddings.
        dropout (float): Dropout rate used during training.

    TODO:
        - Investigate and resolve performance degradation
          when making the number of layers parametric.
          (Note: Performance drops even when keeping layer count
          same as fixed version.)
        - Make the number of layers parametric (configurable ).
    """
    def __init__(self, in_channels, hidden_channels,
                 out_channels, dropout):
        super(GCN, self).__init__()

        # First GCN layer and batch normalization
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        # Second GCN layer and batch normalization
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        # Third (final) GCN layer for output embeddings
        self.conv3 = GCNConv(hidden_channels, out_channels)

        # Dropout rate
        self.dropout = dropout

    def forward(self, data):
        """
        Forward pass of GCN.

        Args:
            data (torch_geometric.data.Data): Input graph data containing node features,
            edge indices, and edge weights.

        Returns:
            torch.Tensor: Normalized node embeddings.
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # First GCN layer with ReLU and dropout
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GCN layer with ReLU and dropout
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third (final) GCN layer and output normalization
        x = self.conv3(x, edge_index, edge_weight)
        x = F.normalize(x, p=2, dim=1)  # Normalize embeddings to unit length

        return x
