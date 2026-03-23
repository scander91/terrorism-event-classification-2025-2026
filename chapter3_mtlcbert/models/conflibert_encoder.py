#!/usr/bin/env python3
"""
ConfliBERT Encoder with Self-Attention Aggregation
====================================================
Encodes enriched text using ConfliBERT (conflict-domain BERT)
and applies multi-head self-attention to aggregate token
representations into a fixed-size event embedding.

For multi-field inputs (text [SEP] attribute1 [SEP] attribute2 ...),
self-attention learns field importance weights automatically.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SelfAttentionAggregation(nn.Module):
    """Multi-head self-attention for aggregating token embeddings."""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len), 1 for valid tokens

        Returns:
            aggregated: (batch, hidden_dim)
        """
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        attn_out, attn_weights = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=key_padding_mask,
        )
        attn_out = self.layer_norm(attn_out + hidden_states)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            aggregated = (attn_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            aggregated = attn_out.mean(dim=1)

        return aggregated


class ConfliBERTEncoder(nn.Module):
    """
    ConfliBERT-based text encoder with self-attention aggregation.

    Input:  Enriched text (summary [SEP] φ(attr1) [SEP] φ(attr2) ...)
    Output: Event embedding ∈ R^768
    """

    def __init__(self, model_name="snowood1/ConfliBERT-scr-uncased",
                 num_heads=8, dropout=0.1, freeze_layers=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

        if freeze_layers > 0:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_layers):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = False

        self.aggregation = SelfAttentionAggregation(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
        )
        self.output_dim = hidden_dim

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            text_embedding: (batch, 768)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        text_embedding = self.aggregation(hidden_states, attention_mask)
        return text_embedding

    @staticmethod
    def get_tokenizer(model_name="snowood1/ConfliBERT-scr-uncased"):
        return AutoTokenizer.from_pretrained(model_name)
