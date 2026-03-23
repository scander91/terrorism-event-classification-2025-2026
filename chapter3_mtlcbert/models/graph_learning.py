#!/usr/bin/env python3
"""
Contextualized Graph Learning Module
======================================
Constructs a heterogeneous event graph capturing relationships
between terrorism incidents through shared entities (perpetrators,
targets, weapons, locations) with temporal decay edge weighting.

Edge weight: w(e_i, e_j) = exp(-λ · |t_i - t_j|)
where λ = 0.01 (temporal decay parameter)

Graph convolution uses GAT (Graph Attention Network) layers
to propagate information across related events.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


class TemporalEdgeBuilder:
    """
    Builds heterogeneous edges between events sharing entities,
    weighted by temporal proximity.
    """

    def __init__(self, decay_lambda=0.01, max_edges_per_node=50):
        self.decay_lambda = decay_lambda
        self.max_edges_per_node = max_edges_per_node

    def build_edges(self, df, shared_columns=None):
        """
        Build edge index and weights from shared entities.

        Args:
            df: DataFrame with columns for shared entities and a date column
            shared_columns: List of columns defining shared relationships
                           (e.g., ["gname", "targtype1_txt", "weaptype1_txt", "region_txt"])

        Returns:
            edge_index: (2, num_edges) tensor
            edge_weight: (num_edges,) tensor
        """
        if shared_columns is None:
            shared_columns = ["gname", "targtype1_txt", "weaptype1_txt", "region_txt"]

        dates = self._extract_dates(df)
        entity_to_events = defaultdict(list)

        for idx, row in df.iterrows():
            for col in shared_columns:
                val = row.get(col, None)
                if val is not None and str(val).strip() and str(val).lower() != "unknown":
                    entity_to_events[(col, str(val))].append(idx)

        src_list, dst_list, weight_list = [], [], []
        edge_count = defaultdict(int)

        for entity_key, event_indices in entity_to_events.items():
            if len(event_indices) < 2 or len(event_indices) > 10000:
                continue

            for i in range(len(event_indices)):
                if edge_count[event_indices[i]] >= self.max_edges_per_node:
                    continue
                for j in range(i + 1, len(event_indices)):
                    if edge_count[event_indices[j]] >= self.max_edges_per_node:
                        continue

                    ei, ej = event_indices[i], event_indices[j]
                    dt = abs(dates[ei] - dates[ej])
                    weight = np.exp(-self.decay_lambda * dt)

                    if weight < 0.01:
                        continue

                    src_list.extend([ei, ej])
                    dst_list.extend([ej, ei])
                    weight_list.extend([weight, weight])
                    edge_count[ei] += 1
                    edge_count[ej] += 1

        if not src_list:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weight = torch.tensor(weight_list, dtype=torch.float)

        return edge_index, edge_weight

    def _extract_dates(self, df):
        """Extract day-level timestamps from GTD date columns."""
        dates = {}
        for idx, row in df.iterrows():
            year = row.get("iyear", 2000)
            month = max(1, int(row.get("imonth", 1)))
            day = max(1, int(row.get("iday", 1)))
            dates[idx] = year * 365 + month * 30 + day
        return dates


class GraphAttentionLayer(nn.Module):
    """Single GAT layer with edge weight support."""

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        if HAS_TORCH_GEOMETRIC:
            self.conv = GATConv(in_dim, out_dim // num_heads,
                               heads=num_heads, dropout=dropout,
                               edge_dim=1)
        else:
            self.linear = nn.Linear(in_dim, out_dim)
            self.attention = nn.Linear(2 * out_dim, 1)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.has_pyg = HAS_TORCH_GEOMETRIC

    def forward(self, x, edge_index, edge_weight=None):
        if self.has_pyg:
            if edge_weight is not None:
                edge_attr = edge_weight.unsqueeze(-1)
            else:
                edge_attr = None
            out = self.conv(x, edge_index, edge_attr=edge_attr)
        else:
            out = self.linear(x)

        out = self.norm(out)
        out = F.elu(out)
        out = self.dropout(out)
        return out


class GraphLearningModule(nn.Module):
    """
    Heterogeneous graph neural network for event representation.

    Input:  Initial node features (from text encoder or structured features)
    Output: Graph-enhanced event embeddings ∈ R^{graph_dim}
    """

    def __init__(self, input_dim, hidden_dim=256, output_dim=256,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else output_dim
            self.layers.append(
                GraphAttentionLayer(in_d, out_d, num_heads, dropout))

        self.output_dim = output_dim

    def forward(self, node_features, edge_index, edge_weight=None, batch=None):
        """
        Args:
            node_features: (num_nodes, input_dim)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,)
            batch: (num_nodes,) batch assignment for pooling

        Returns:
            graph_embeddings: (batch_size, output_dim) if batch provided
                              (num_nodes, output_dim) otherwise
        """
        x = self.input_proj(node_features)

        for layer in self.layers:
            residual = x
            x = layer(x, edge_index, edge_weight)
            if x.shape == residual.shape:
                x = x + residual

        if batch is not None and HAS_TORCH_GEOMETRIC:
            x = global_mean_pool(x, batch)

        return x
