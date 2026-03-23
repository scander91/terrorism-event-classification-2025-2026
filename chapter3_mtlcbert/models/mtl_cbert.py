#!/usr/bin/env python3
"""
MTL-CBERT: Multi-Task Learning Contextualized BERT
=====================================================
Graph-augmented multi-task classification framework for terrorism
attack type classification across nine categories.

Architecture:
    1. Feature Enrichment → attribute-to-text templates (Cramér's V > 0.25)
    2. ConfliBERT Encoder → self-attention aggregation → h_text ∈ R^768
    3. Graph Learning → heterogeneous graph + temporal decay → h_graph ∈ R^256
    4. Gated Fusion → adaptive combination → h_fused ∈ R^512
    5. Focal Loss (γ=2.0) + inverse-frequency class weighting

Loss:
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conflibert_encoder import ConfliBERTEncoder
from .graph_learning import GraphLearningModule
from .gated_fusion import GatedFusion


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weight tensor (inverse frequency)
        reduction: 'mean' or 'sum'
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MTLCBERT(nn.Module):
    """
    Full MTL-CBERT model.

    Combines ConfliBERT text encoding, heterogeneous graph learning
    with temporal decay, and gated fusion for attack type classification.

    Args:
        num_classes: Number of attack type categories (9)
        conflibert_model: HuggingFace model identifier
        graph_input_dim: Initial node feature dimension
        graph_hidden_dim: Graph hidden layer dimension
        graph_output_dim: Graph output embedding dimension
        fusion_dim: Dimension after gated fusion
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_classes=9,
        conflibert_model="snowood1/ConfliBERT-scr-uncased",
        graph_input_dim=768,
        graph_hidden_dim=256,
        graph_output_dim=256,
        fusion_dim=512,
        dropout=0.3,
        graph_num_heads=4,
        graph_num_layers=2,
    ):
        super().__init__()

        self.text_encoder = ConfliBERTEncoder(
            model_name=conflibert_model,
            num_heads=8,
            dropout=dropout,
        )

        self.graph_module = GraphLearningModule(
            input_dim=graph_input_dim,
            hidden_dim=graph_hidden_dim,
            output_dim=graph_output_dim,
            num_heads=graph_num_heads,
            num_layers=graph_num_layers,
            dropout=dropout,
        )

        self.fusion = GatedFusion(
            text_dim=self.text_encoder.output_dim,
            graph_dim=graph_output_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask,
                graph_node_features=None, edge_index=None,
                edge_weight=None, batch_graph=None):
        """
        Args:
            input_ids: (batch, seq_len) tokenized enriched text
            attention_mask: (batch, seq_len)
            graph_node_features: (num_nodes, feat_dim) node features
            edge_index: (2, num_edges) graph edges
            edge_weight: (num_edges,) temporal decay weights
            batch_graph: (num_nodes,) batch assignment

        Returns:
            logits: (batch, num_classes)
        """
        text_emb = self.text_encoder(input_ids, attention_mask)

        if graph_node_features is not None and edge_index is not None:
            graph_emb = self.graph_module(
                graph_node_features, edge_index, edge_weight, batch_graph)

            if graph_emb.shape[0] != text_emb.shape[0]:
                graph_emb = graph_emb[:text_emb.shape[0]]

            fused = self.fusion(text_emb, graph_emb)
        else:
            proj = nn.Linear(text_emb.shape[-1], self.fusion.output_dim).to(text_emb.device)
            fused = proj(text_emb)

        logits = self.classifier(fused)
        return logits

    def get_text_embeddings(self, input_ids, attention_mask):
        """Extract text-only embeddings (for graph node initialization)."""
        with torch.no_grad():
            return self.text_encoder(input_ids, attention_mask)
