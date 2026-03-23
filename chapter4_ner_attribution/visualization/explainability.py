# ======================================================================
# TerrorismNER: Explainability Module
# SHAP, LIME, and Attention Visualization
# ======================================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F

# ======================================================================
# SHAP EXPLAINER FOR NER
# ======================================================================

class NERSHAPExplainer:
    """SHAP explanations for NER models."""
    
    def __init__(self, model, tokenizer, label2id, id2label, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
        self.device = device
        
    def explain(self, text: str, target_token_idx: int = None) -> Dict:
        """
        Generate SHAP-like importance scores for tokens.
        
        Uses occlusion-based importance estimation.
        """
        self.model.eval()
        
        # Tokenize
        tokens = text.split()
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get baseline prediction
        with torch.no_grad():
            # Create dummy gazetteer_ids
            gazetteer_ids = torch.zeros_like(input_ids)
            output = self.model(input_ids, attention_mask, gazetteer_ids)
            baseline_logits = output['emissions']
            baseline_probs = F.softmax(baseline_logits, dim=-1)
        
        # Calculate importance for each token
        importances = []
        word_ids = encoding.word_ids()
        
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            
            # Mask this token
            masked_input = input_ids.clone()
            masked_input[0, i] = self.tokenizer.mask_token_id or self.tokenizer.pad_token_id
            
            with torch.no_grad():
                masked_output = self.model(masked_input, attention_mask, gazetteer_ids)
                masked_probs = F.softmax(masked_output['emissions'], dim=-1)
            
            # Calculate importance as probability change
            if target_token_idx is not None:
                importance = (baseline_probs[0, target_token_idx] - 
                            masked_probs[0, target_token_idx]).abs().sum().item()
            else:
                importance = (baseline_probs - masked_probs).abs().sum().item()
            
            importances.append({
                'token_idx': i,
                'word_idx': word_id,
                'token': self.tokenizer.decode([input_ids[0, i].item()]),
                'importance': importance
            })
        
        # Aggregate by word
        word_importances = defaultdict(float)
        for imp in importances:
            word_importances[imp['word_idx']] += imp['importance']
        
        return {
            'tokens': tokens,
            'word_importances': dict(word_importances),
            'token_importances': importances
        }
    
    def visualize(self, explanation: Dict, save_path: str = None):
        """Visualize SHAP-like importance scores."""
        
        tokens = explanation['tokens']
        importances = [explanation['word_importances'].get(i, 0) for i in range(len(tokens))]
        
        # Normalize
        max_imp = max(importances) if importances else 1
        norm_importances = [imp / max_imp for imp in importances]
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        # Create colored text representation
        colors = plt.cm.RdYlGn_r(norm_importances)
        
        for i, (token, color, imp) in enumerate(zip(tokens, colors, norm_importances)):
            ax.bar(i, imp, color=color, edgecolor='black', linewidth=0.5)
            ax.text(i, -0.1, token, ha='center', va='top', fontsize=9, rotation=45)
        
        ax.set_ylabel('Importance Score')
        ax.set_title('Token Importance (SHAP-like)')
        ax.set_xlim(-0.5, len(tokens) - 0.5)
        ax.set_xticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ======================================================================
# LIME EXPLAINER FOR NER
# ======================================================================

class NERLIMEExplainer:
    """LIME-like explanations for NER models."""
    
    def __init__(self, model, tokenizer, label2id, id2label, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
        self.device = device
        
    def explain(self, text: str, num_samples: int = 100, 
                target_label: str = 'B-TERROR_GROUP') -> Dict:
        """
        Generate LIME explanation by sampling perturbations.
        """
        self.model.eval()
        
        tokens = text.split()
        n_tokens = len(tokens)
        
        # Generate perturbations
        perturbations = []
        predictions = []
        
        for _ in range(num_samples):
            # Random mask
            mask = np.random.binomial(1, 0.8, n_tokens)  # Keep 80% of tokens
            
            # Create perturbed text
            perturbed_tokens = [t if m else '[MASK]' for t, m in zip(tokens, mask)]
            perturbed_text = ' '.join(perturbed_tokens)
            
            # Get prediction
            encoding = self.tokenizer(
                perturbed_tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            gazetteer_ids = torch.zeros_like(input_ids)
            
            with torch.no_grad():
                output = self.model(input_ids, attention_mask, gazetteer_ids)
                probs = F.softmax(output['emissions'], dim=-1)
            
            # Get probability of target label
            target_idx = self.label2id.get(target_label, 1)
            avg_prob = probs[0, :sum(mask), target_idx].mean().item()
            
            perturbations.append(mask)
            predictions.append(avg_prob)
        
        # Fit linear model
        X = np.array(perturbations)
        y = np.array(predictions)
        
        # Simple linear regression for feature importance
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        importances = model.coef_
        
        return {
            'tokens': tokens,
            'importances': importances.tolist(),
            'intercept': model.intercept_,
            'target_label': target_label
        }
    
    def visualize(self, explanation: Dict, save_path: str = None, top_k: int = 15):
        """Visualize LIME importance scores."""
        
        tokens = explanation['tokens']
        importances = explanation['importances']
        
        # Sort by absolute importance
        sorted_idx = np.argsort(np.abs(importances))[::-1][:top_k]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        selected_tokens = [tokens[i] for i in sorted_idx]
        selected_imps = [importances[i] for i in sorted_idx]
        
        colors = ['green' if imp > 0 else 'red' for imp in selected_imps]
        
        y_pos = np.arange(len(selected_tokens))
        ax.barh(y_pos, selected_imps, color=colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(selected_tokens)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'LIME Feature Importance ({explanation["target_label"]})')
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ======================================================================
# ATTENTION VISUALIZATION
# ======================================================================

class AttentionVisualizer:
    """Visualize attention weights from transformer models."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def get_attention_weights(self, text: str) -> Dict:
        """Extract attention weights from model."""
        
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Extract attentions from all layers
        attentions = outputs.attentions  # List of (batch, heads, seq, seq)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Average over layers and heads
        avg_attention = torch.stack(attentions).mean(dim=[0, 1, 2])  # (seq, seq)
        avg_attention = avg_attention.cpu().numpy()
        
        return {
            'tokens': tokens,
            'attention': avg_attention,
            'all_attentions': [a.cpu().numpy() for a in attentions]
        }
    
    def visualize_attention_heatmap(self, attention_data: Dict, 
                                     save_path: str = None,
                                     max_tokens: int = 30):
        """Visualize attention as heatmap."""
        
        tokens = attention_data['tokens'][:max_tokens]
        attention = attention_data['attention'][:max_tokens, :max_tokens]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            ax=ax,
            square=True,
            cbar_kws={'shrink': 0.8}
        )
        
        ax.set_title('Attention Weights (Average)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_attention_by_layer(self, attention_data: Dict,
                                      save_path: str = None):
        """Visualize attention patterns across layers."""
        
        all_attentions = attention_data['all_attentions']
        n_layers = len(all_attentions)
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for layer_idx, attention in enumerate(all_attentions[:12]):
            ax = axes[layer_idx]
            
            # Average over batch and heads
            avg_attn = attention[0].mean(axis=0)[:20, :20]
            
            sns.heatmap(avg_attn, cmap='Blues', ax=ax, cbar=False, square=True)
            ax.set_title(f'Layer {layer_idx + 1}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('Attention Patterns Across Layers', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ======================================================================
# CONFUSION MATRIX VISUALIZATION
# ======================================================================

def plot_confusion_matrix(y_true: List, y_pred: List, labels: List[str],
                          save_path: str = None, normalize: bool = True):
    """Plot confusion matrix for NER or classification."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        square=True
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ======================================================================
# ERROR ANALYSIS
# ======================================================================

class NERErrorAnalyzer:
    """Analyze errors in NER predictions."""
    
    def __init__(self, id2label: Dict[int, str]):
        self.id2label = id2label
        self.errors = []
        
    def collect_errors(self, texts: List[str], predictions: List[List[str]], 
                       references: List[List[str]]):
        """Collect and categorize errors."""
        
        for text, preds, refs in zip(texts, predictions, references):
            tokens = text.split()
            
            for i, (pred, ref) in enumerate(zip(preds, refs)):
                if pred != ref:
                    error_type = self._categorize_error(pred, ref)
                    
                    self.errors.append({
                        'text': text,
                        'token': tokens[i] if i < len(tokens) else '',
                        'position': i,
                        'predicted': pred,
                        'reference': ref,
                        'error_type': error_type
                    })
    
    def _categorize_error(self, pred: str, ref: str) -> str:
        """Categorize error type."""
        
        if pred == 'O' and ref.startswith('B-'):
            return 'missed_entity_start'
        elif pred == 'O' and ref.startswith('I-'):
            return 'missed_entity_continuation'
        elif pred.startswith('B-') and ref == 'O':
            return 'false_positive_start'
        elif pred.startswith('I-') and ref == 'O':
            return 'false_positive_continuation'
        elif pred.startswith('B-') and ref.startswith('I-'):
            return 'wrong_boundary'
        elif pred.startswith('I-') and ref.startswith('B-'):
            return 'wrong_boundary'
        else:
            return 'type_error'
    
    def get_error_statistics(self) -> Dict:
        """Get error statistics."""
        
        stats = defaultdict(int)
        for error in self.errors:
            stats[error['error_type']] += 1
        
        total = len(self.errors)
        percentages = {k: v / total * 100 for k, v in stats.items()} if total > 0 else {}
        
        return {
            'counts': dict(stats),
            'percentages': percentages,
            'total_errors': total
        }
    
    def visualize_errors(self, save_path: str = None):
        """Visualize error distribution."""
        
        stats = self.get_error_statistics()
        
        if not stats['counts']:
            print("No errors to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        ax1 = axes[0]
        error_types = list(stats['counts'].keys())
        counts = list(stats['counts'].values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(error_types)))
        
        ax1.bar(error_types, counts, color=colors, edgecolor='black')
        ax1.set_ylabel('Count')
        ax1.set_title('Error Distribution')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2 = axes[1]
        ax2.pie(counts, labels=error_types, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Error Type Proportions')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_example_errors(self, error_type: str, n: int = 5) -> List[Dict]:
        """Get example errors of a specific type."""
        
        examples = [e for e in self.errors if e['error_type'] == error_type]
        return examples[:n]


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("="*60)
    print("TerrorismNER: Explainability Module")
    print("="*60)
    
    print("\nThis module provides:")
    print("1. SHAP-like token importance analysis")
    print("2. LIME-based explanations")
    print("3. Attention visualization")
    print("4. Confusion matrix visualization")
    print("5. Error analysis")
    
    print("\nUsage example:")
    print("""
    from explainability import NERSHAPExplainer, NERLIMEExplainer
    
    # Initialize explainer
    explainer = NERSHAPExplainer(model, tokenizer, label2id, id2label)
    
    # Explain a prediction
    explanation = explainer.explain("Taliban attacked the city")
    
    # Visualize
    explainer.visualize(explanation, save_path='shap_explanation.png')
    """)


if __name__ == "__main__":
    main()
