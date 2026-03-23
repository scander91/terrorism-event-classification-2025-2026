#!/usr/bin/env python3
"""
Gated Fusion Mechanism
========================
Adaptively combines text embeddings (R^768) and graph embeddings
(R^256) using a learned gating function.

The gate vector g ∈ [0,1]^d controls the contribution of each
modality at each dimension:

    g = σ(W_g · [h_text; h_graph] + b_g)
    h_fused = g ⊙ W_t(h_text) + (1 - g) ⊙ W_g(h_graph)

where σ is the sigmoid function and ⊙ is element-wise multiplication.
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Gated multimodal fusion for text and graph representations.

    Args:
        text_dim: Dimension of text embeddings (768 for ConfliBERT)
        graph_dim: Dimension of graph embeddings (256)
        fusion_dim: Output dimension after fusion (512)
        dropout: Dropout rate
    """

    def __init__(self, text_dim=768, graph_dim=256, fusion_dim=512, dropout=0.3):
        super().__init__()

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gate = nn.Sequential(
            nn.Linear(text_dim + graph_dim, fusion_dim),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_dim = fusion_dim

    def forward(self, text_emb, graph_emb):
        """
        Args:
            text_emb: (batch, text_dim) from ConfliBERT encoder
            graph_emb: (batch, graph_dim) from graph learning module

        Returns:
            fused: (batch, fusion_dim)
        """
        h_text = self.text_proj(text_emb)
        h_graph = self.graph_proj(graph_emb)

        concat = torch.cat([text_emb, graph_emb], dim=-1)
        g = self.gate(concat)

        fused = g * h_text + (1 - g) * h_graph
        fused = self.output_proj(fused)

        return fused
