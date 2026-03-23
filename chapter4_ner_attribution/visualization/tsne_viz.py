#!/usr/bin/env python3
"""
===============================================================================
t-SNE Visualization for Terrorist Group Classification (FIXED)
===============================================================================
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path('/home/macierz/mohabdal/TerrorismNER_Project')
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
FIGURES_DIR = BASE_DIR / 'figures' / 'phase2'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Colors for up to 20 groups
COLORS_20 = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
]

# =============================================================================
# DATA LOADING (FIXED)
# =============================================================================
def load_classification_data(num_groups):
    """Load classification data for specified number of groups."""
    data_path = CHECKPOINT_DIR / f'classification_data_{num_groups}.pkl'
    if not data_path.exists():
        print(f"Warning: {data_path} not found")
        return None, None
    
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    
    # Use correct column names
    texts = df['clean_text'].tolist()
    labels = df['canonical_group'].tolist()
    
    print(f"Loaded {num_groups} groups: {len(texts)} samples")
    return texts, labels

# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================
def extract_embeddings(texts, labels, model_name='roberta-base', max_samples=500, max_length=256):
    """Extract [CLS] embeddings from transformer model."""
    
    print(f"Extracting embeddings from {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    
    # Sample if too many
    if len(texts) > max_samples:
        indices = np.random.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    embeddings = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting"):
            inputs = tokenizer(
                str(text), 
                return_tensors='pt', 
                truncation=True, 
                max_length=max_length,
                padding='max_length'
            ).to(DEVICE)
            
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding[0])
    
    return np.array(embeddings), labels

# =============================================================================
# t-SNE COMPUTATION
# =============================================================================
def compute_tsne(embeddings, perplexity=30, max_iter=1000, random_state=42):
    """Compute t-SNE dimensionality reduction."""
    print(f"Computing t-SNE (perplexity={perplexity})...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        max_iter=max_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    
    return tsne.fit_transform(embeddings)

# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_tsne_single(embeddings_2d, labels, title, save_path, num_groups=10):
    """Plot single t-SNE visualization."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    unique_labels = le.classes_
    
    if len(unique_labels) <= 20:
        colors = COLORS_20[:len(unique_labels)]
    else:
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))
        colors = [cmap(i) for i in range(len(unique_labels))]
    
    for i, label in enumerate(unique_labels):
        mask = labels_encoded == i
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i % len(colors)]],
            label=label[:25] + '...' if len(str(label)) > 25 else label,
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if num_groups <= 20:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

def plot_tsne_scalability(results_dict, save_path):
    """Plot t-SNE across scales (10, 20, 50, 100 groups)."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (scale, (embeddings_2d, labels)) in enumerate(sorted(results_dict.items())):
        ax = axes[idx]
        
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        unique_labels = le.classes_
        
        cmap = plt.cm.get_cmap('tab20', min(20, len(unique_labels)))
        
        for i in range(len(unique_labels)):
            mask = labels_encoded == i
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[cmap(i % 20)],
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
        ax.set_title(f'{scale} Groups', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.text(0.98, 0.98, f'n={len(labels)}', transform=ax.transAxes, fontsize=10,
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('t-SNE Visualization: Classification Scalability', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("t-SNE VISUALIZATION FOR TERRORIST GROUP CLASSIFICATION")
    print("=" * 70)
    
    # Figure 1: 10 groups
    print("\n[1/2] Generating t-SNE for 10 groups...")
    texts, labels = load_classification_data(10)
    if texts:
        embeddings, labels = extract_embeddings(texts, labels, max_samples=500)
        embeddings_2d = compute_tsne(embeddings, perplexity=30)
        plot_tsne_single(embeddings_2d, labels,
                        't-SNE: Top 10 Terrorist Groups (RoBERTa)',
                        FIGURES_DIR / 'tsne_10_groups.png', num_groups=10)
    
    # Figure 2: Scalability
    print("\n[2/2] Generating t-SNE scalability comparison...")
    scalability_results = {}
    
    for num_groups in [10, 20, 50, 100]:
        texts, labels = load_classification_data(num_groups)
        if texts:
            max_samples = 400 if num_groups <= 20 else 300
            embeddings, labels = extract_embeddings(texts, labels, max_samples=max_samples)
            perplexity = min(30, len(embeddings) // 5)
            embeddings_2d = compute_tsne(embeddings, perplexity=perplexity)
            scalability_results[num_groups] = (embeddings_2d, labels)
    
    if scalability_results:
        plot_tsne_scalability(scalability_results, FIGURES_DIR / 'tsne_scalability_comparison.png')
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
