# ======================================================================
# TerrorismNER: Data Processing & Analysis (CORRECTED VERSION)
# ======================================================================
#
# هذا الملف المصحح يتضمن:
# 1. توحيد أسماء المجموعات بشكل صحيح
# 2. تحليل شامل للبيانات
# 3. إعداد بيانات NER و Classification
# 4. حفظ تلقائي وإدارة الذاكرة
#
# ======================================================================

import os
import sys
import gc
import re
import json
import pickle
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ======================================================================
# CONFIGURATION
# ======================================================================

class Config:
    """Central configuration for the entire project."""
    
    PROJECT_NAME = "TerrorismNER_Project"
    BASE_DIR = Path(f"/home/macierz/mohabdal/TerrorismNER_Project")
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = BASE_DIR / "figures"
    CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
    LOGS_DIR = BASE_DIR / "logs"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Data settings
    GTD_PATH = "/home/macierz/mohabdal/TerrorismNER_Project/globalterrorismdb_0522dist (1).xlsx"
    TEXT_COLUMNS = ['summary', 'motive', 'addnotes', 'scite1', 'scite2', 'scite3']
    GROUP_COLUMN = 'gname'
    
    # Model settings
    SEED = 42
    MAX_SEQ_LENGTH = 256
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    
    # Experiment settings
    GROUP_SIZES = [10, 20, 50, 'all']
    MAX_SAMPLES_PER_GROUP = 500
    MIN_SAMPLES_PER_GROUP = 10
    
    # NER Labels
    NER_LABELS_BASIC = ['O', 'B-TERROR_GROUP', 'I-TERROR_GROUP']
    NER_LABELS_EXTENDED = [
        'O', 
        'B-TERROR_GROUP', 'I-TERROR_GROUP',
        'B-LOC', 'I-LOC',
        'B-WEAPON', 'I-WEAPON',
        'B-ATTACK_TYPE', 'I-ATTACK_TYPE'
    ]
    
    @classmethod
    def setup_directories(cls):
        """Create all necessary directories."""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, 
                         cls.FIGURES_DIR, cls.CHECKPOINTS_DIR, cls.LOGS_DIR, cls.CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Project directories created at: {cls.BASE_DIR}")
    
    @classmethod
    def set_seeds(cls):
        """Set all random seeds for reproducibility."""
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        try:
            import torch
            torch.manual_seed(cls.SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cls.SEED)
        except ImportError:
            pass
        print(f"✅ Random seeds set to: {cls.SEED}")


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

def save_checkpoint(obj, name: str, config: Config = Config):
    """Save object to checkpoint."""
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CHECKPOINTS_DIR / f"{name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"💾 Checkpoint saved: {path}")

def load_checkpoint(name: str, config: Config = Config):
    """Load object from checkpoint."""
    path = config.CHECKPOINTS_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"📂 Checkpoint loaded: {path}")
        return obj
    return None

def save_figure(fig, name: str, config: Config = Config):
    """Save figure in multiple formats."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ['png', 'pdf']:
        path = config.FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"📊 Figure saved: {name}")
    plt.close(fig)

def log_message(message: str, config: Config = Config):
    """Log message to file and print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / "experiment_log.txt"
    with open(log_file, 'a') as f:
        f.write(log_entry + "\n")
    print(log_entry)


# ======================================================================
# IMPORT GROUP NORMALIZATION
# ======================================================================

try:
    from group_normalization import (
        TERRORISM_GAZETTEER, GAZETTEER_FLAT, 
        GroupNameNormalizer, validate_gazetteer
    )
    print("✅ Group normalization module loaded")
except ImportError:
    print("⚠️ Group normalization module not found, using inline version")
    
    # Inline minimal gazetteer
    TERRORISM_GAZETTEER = {
        "Taliban": ["Taliban", "Afghan Taliban", "Taleban"],
        "Al-Qaeda": ["Al-Qaeda", "Al-Qaida", "Al Qaeda", "AQ"],
        "Islamic State": ["Islamic State", "ISIS", "ISIL", "Daesh", "IS"],
        "Unknown": ["Unknown", "unknown"]
    }
    
    GAZETTEER_FLAT = {}
    for canonical, aliases in TERRORISM_GAZETTEER.items():
        for alias in aliases:
            GAZETTEER_FLAT[alias.lower()] = canonical


# ======================================================================
# WEAPON & ATTACK TYPE GAZETTEERS
# ======================================================================

WEAPON_GAZETTEER = {
    "Explosives": ["bomb", "explosive", "ied", "improvised explosive device", "car bomb", 
                   "suicide bomb", "truck bomb", "dynamite", "grenade", "mortar", "rocket",
                   "landmine", "mine", "detonator"],
    "Firearms": ["gun", "rifle", "ak-47", "ak47", "kalashnikov", "pistol", "machine gun",
                 "assault rifle", "automatic weapon", "sniper", "firearm", "shooting",
                 "gunshot", "bullets", "ammunition"],
    "Incendiary": ["fire", "arson", "molotov", "incendiary", "firebomb", "petrol bomb",
                   "burning", "flames"],
    "Melee": ["knife", "machete", "axe", "sword", "stabbing", "blade", "cutting"],
    "Chemical": ["chemical", "poison", "gas", "nerve agent", "sarin", "vx", "toxic"],
    "Vehicle": ["vehicle", "car ramming", "truck attack", "vehicular", "ramming"],
}

ATTACK_TYPE_GAZETTEER = {
    "Bombing/Explosion": ["bombing", "explosion", "blast", "detonation", "blew up", "exploded"],
    "Armed Assault": ["shooting", "gunfire", "armed assault", "opened fire", "ambush", "raid",
                      "firefight", "gunned down"],
    "Assassination": ["assassination", "assassinated", "targeted killing", "executed"],
    "Kidnapping": ["kidnapping", "kidnapped", "hostage", "abduction", "abducted", "hijacking",
                   "seized", "captured"],
    "Facility Attack": ["sabotage", "infrastructure", "facility attack", "destroyed building"],
}

def create_flat_gazetteer(gazetteer_dict):
    """Create flat dictionary from nested gazetteer."""
    flat = {}
    for canonical, aliases in gazetteer_dict.items():
        for alias in aliases:
            flat[alias.lower()] = canonical
    return flat

WEAPON_FLAT = create_flat_gazetteer(WEAPON_GAZETTEER)
ATTACK_TYPE_FLAT = create_flat_gazetteer(ATTACK_TYPE_GAZETTEER)


# ======================================================================
# DATA LOADING AND PREPROCESSING
# ======================================================================

class DataProcessor:
    """
    Handles all data loading and preprocessing operations.
    
    Features:
    - Automatic group name normalization
    - Text combining from multiple columns
    - Entity detection for NER
    - Proper train/val/test splits
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.df = None
        self.normalizer = None
        self.stats = {}
        
        # Try to load GroupNameNormalizer
        try:
            from group_normalization import GroupNameNormalizer
            self.normalizer = GroupNameNormalizer()
        except ImportError:
            self.normalizer = None
        
    def load_gtd(self, path: str = None) -> pd.DataFrame:
        """Load GTD Excel file."""
        path = path or self.config.GTD_PATH
        
        log_message(f"Loading GTD data from: {path}")
        
        # Check cache
        cache_path = self.config.CACHE_DIR / "gtd_raw.pkl"
        if cache_path.exists():
            self.df = pd.read_pickle(cache_path)
            log_message(f"Loaded from cache: {self.df.shape}")
            return self.df
        
        # Load from Excel
        self.df = pd.read_excel(path)
        
        # Cache it
        self.config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.df.to_pickle(cache_path)
        log_message(f"GTD loaded: {self.df.shape}")
        
        return self.df
    
    def preprocess(self) -> pd.DataFrame:
        """Preprocess the GTD data with proper group normalization."""
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_gtd() first.")
        
        log_message("Starting preprocessing...")
        
        # 1. Get available text columns
        available_text_cols = [col for col in self.config.TEXT_COLUMNS if col in self.df.columns]
        log_message(f"Available text columns: {available_text_cols}")
        
        # 2. Combine text columns
        def combine_text(row):
            texts = []
            for col in available_text_cols:
                if pd.notna(row[col]) and str(row[col]).strip():
                    text = str(row[col]).strip()
                    texts.append(text)
            return ' '.join(texts)
        
        log_message("Combining text columns...")
        self.df['combined_text'] = self.df.apply(combine_text, axis=1)
        
        # 3. Clean text
        def clean_text(text):
            if pd.isna(text) or not isinstance(text, str):
                return ""
            text = str(text).strip()
            # Remove URLs
            text = re.sub(r'http\S+|www\.\S+', '', text)
            # Remove emails
            text = re.sub(r'\S+@\S+', '', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        self.df['clean_text'] = self.df['combined_text'].apply(clean_text)
        
        # 4. Clean group names
        def clean_group_name(name):
            if pd.isna(name) or not isinstance(name, str):
                return ""
            name = str(name).strip()
            # Remove parenthetical info for matching
            name_clean = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
            return name_clean
        
        self.df['clean_group'] = self.df[self.config.GROUP_COLUMN].apply(clean_group_name)
        
        # 5. Normalize group names using the comprehensive normalizer
        log_message("Normalizing group names...")
        
        if self.normalizer:
            self.df['canonical_group'] = self.df['clean_group'].apply(
                lambda x: self.normalizer.normalize(x, log=True)
            )
            
            # Get normalization report
            report = self.normalizer.get_normalization_report()
            log_message(f"Normalization report: {report['methods_used']}")
            
            if report['unknown_count'] > 0:
                log_message(f"⚠️ Found {report['unknown_count']} unknown groups")
                # Save unknown groups for review
                unknown = self.normalizer.get_unknown_groups_sorted()
                with open(self.config.RESULTS_DIR / "unknown_groups.json", 'w') as f:
                    json.dump(unknown[:100], f, indent=2)  # Top 100
        else:
            # Fallback to simple normalization
            def simple_normalize(name):
                name_lower = name.lower().strip()
                if name_lower in GAZETTEER_FLAT:
                    return GAZETTEER_FLAT[name_lower]
                # Try partial match
                for alias, canonical in GAZETTEER_FLAT.items():
                    if alias in name_lower or name_lower in alias:
                        return canonical
                return name
            
            self.df['canonical_group'] = self.df['clean_group'].apply(simple_normalize)
        
        # 6. Filter valid records
        initial_count = len(self.df)
        
        # Remove empty text
        self.df = self.df[self.df['clean_text'].str.len() >= 50]
        log_message(f"After text filter: {len(self.df)} (removed {initial_count - len(self.df)})")
        
        # Remove empty groups
        self.df = self.df[self.df['clean_group'] != '']
        
        # Remove unknown groups
        self.df = self.df[self.df['clean_group'].str.lower() != 'unknown']
        self.df = self.df[self.df['canonical_group'] != 'Unknown']
        
        log_message(f"After preprocessing: {len(self.df)} records")
        
        # 7. Save statistics
        self.stats = {
            'total_records': len(self.df),
            'unique_groups': self.df['canonical_group'].nunique(),
            'group_distribution': self.df['canonical_group'].value_counts().to_dict()
        }
        
        # Save checkpoint
        save_checkpoint(self.df, "preprocessed_df", self.config)
        
        return self.df
    
    def analyze_group_normalization(self) -> pd.DataFrame:
        """Analyze group name normalization results."""
        
        if self.df is None:
            raise ValueError("Data not preprocessed yet")
        
        # Compare original vs canonical
        comparison = self.df.groupby(['clean_group', 'canonical_group']).size().reset_index(name='count')
        comparison = comparison.sort_values('count', ascending=False)
        
        # Save analysis
        comparison.to_csv(self.config.RESULTS_DIR / "group_normalization_analysis.csv", index=False)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top canonical groups
        ax1 = axes[0]
        top_canonical = self.df['canonical_group'].value_counts().head(20)
        top_canonical.plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_title('Top 20 Canonical Groups')
        ax1.set_xlabel('Number of Records')
        ax1.invert_yaxis()
        
        # Normalization method distribution (if available)
        ax2 = axes[1]
        if self.normalizer:
            report = self.normalizer.get_normalization_report()
            methods = report['methods_used']
            ax2.pie(methods.values(), labels=methods.keys(), autopct='%1.1f%%')
            ax2.set_title('Normalization Methods Used')
        else:
            ax2.text(0.5, 0.5, 'No detailed normalization data', ha='center', va='center')
            ax2.set_title('Normalization Methods')
        
        plt.tight_layout()
        save_figure(fig, 'group_normalization_analysis', self.config)
        
        return comparison
    
    def prepare_ner_data(self, extended: bool = False) -> Tuple[List, List, List]:
        """
        Prepare data for NER task with proper BIO formatting.
        
        Args:
            extended: If True, include LOC, WEAPON, ATTACK_TYPE
            
        Returns:
            train_data, val_data, test_data
        """
        
        log_message(f"Preparing NER data (extended={extended})...")
        
        if self.df is None:
            raise ValueError("Data not preprocessed yet")
        
        # Find entities in text
        bio_data = []
        
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Creating BIO"):
            text = row['clean_text']
            text_lower = text.lower()
            
            if len(text) < 50:
                continue
            
            entities = []
            
            # 1. Find terrorism group
            canonical = row['canonical_group']
            group_found = False
            
            # Try all aliases for this canonical group
            if canonical in TERRORISM_GAZETTEER:
                for alias in TERRORISM_GAZETTEER[canonical]:
                    alias_lower = alias.lower()
                    start = text_lower.find(alias_lower)
                    if start != -1:
                        entities.append({
                            'start': start,
                            'end': start + len(alias),
                            'text': text[start:start + len(alias)],
                            'type': 'TERROR_GROUP',
                            'canonical': canonical
                        })
                        group_found = True
                        break
            
            # Also try the clean group name
            if not group_found:
                clean_group_lower = row['clean_group'].lower()
                start = text_lower.find(clean_group_lower)
                if start != -1:
                    entities.append({
                        'start': start,
                        'end': start + len(clean_group_lower),
                        'text': text[start:start + len(clean_group_lower)],
                        'type': 'TERROR_GROUP',
                        'canonical': canonical
                    })
                    group_found = True
            
            if not group_found:
                continue  # Skip if no group found in text
            
            # 2. Find locations (extended mode)
            if extended:
                for loc_col in ['country_txt', 'city', 'provstate']:
                    if loc_col in row.index and pd.notna(row[loc_col]):
                        loc = str(row[loc_col]).lower()
                        if len(loc) >= 3:
                            start = text_lower.find(loc)
                            if start != -1:
                                entities.append({
                                    'start': start,
                                    'end': start + len(loc),
                                    'text': text[start:start + len(loc)],
                                    'type': 'LOC',
                                    'canonical': row[loc_col]
                                })
                                break
                
                # Find weapons
                for weapon_alias, weapon_type in WEAPON_FLAT.items():
                    if len(weapon_alias) >= 4:
                        start = text_lower.find(weapon_alias)
                        if start != -1:
                            entities.append({
                                'start': start,
                                'end': start + len(weapon_alias),
                                'text': text[start:start + len(weapon_alias)],
                                'type': 'WEAPON',
                                'canonical': weapon_type
                            })
                            break
                
                # Find attack types
                for attack_alias, attack_type in ATTACK_TYPE_FLAT.items():
                    if len(attack_alias) >= 5:
                        start = text_lower.find(attack_alias)
                        if start != -1:
                            entities.append({
                                'start': start,
                                'end': start + len(attack_alias),
                                'text': text[start:start + len(attack_alias)],
                                'type': 'ATTACK_TYPE',
                                'canonical': attack_type
                            })
                            break
            
            # Convert to BIO format
            tokens = text.split()
            labels = ['O'] * len(tokens)
            
            # Create character to token mapping
            char_to_token = {}
            current_pos = 0
            for i, token in enumerate(tokens):
                for j in range(len(token)):
                    char_to_token[current_pos + j] = i
                current_pos += len(token) + 1
            
            # Assign labels
            for entity in entities:
                entity_type = entity['type']
                start, end = entity['start'], entity['end']
                first_token = True
                
                for char_pos in range(start, min(end, len(text))):
                    if char_pos in char_to_token:
                        token_idx = char_to_token[char_pos]
                        if token_idx < len(labels):
                            if labels[token_idx] == 'O':  # Don't overwrite existing labels
                                if first_token:
                                    labels[token_idx] = f'B-{entity_type}'
                                    first_token = False
                                else:
                                    labels[token_idx] = f'I-{entity_type}'
            
            # Verify at least one entity label
            if any(l.startswith('B-') for l in labels):
                bio_data.append({
                    'tokens': tokens,
                    'labels': labels,
                    'canonical_group': canonical,
                    'entities': entities
                })
        
        log_message(f"Created {len(bio_data)} BIO samples")
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        train_data, temp_data = train_test_split(
            bio_data, train_size=0.7, random_state=self.config.SEED
        )
        val_data, test_data = train_test_split(
            temp_data, train_size=0.5, random_state=self.config.SEED
        )
        
        log_message(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Save checkpoint
        suffix = "_extended" if extended else "_basic"
        save_checkpoint({
            'train': train_data,
            'val': val_data,
            'test': test_data
        }, f'ner_data{suffix}', self.config)
        
        return train_data, val_data, test_data
    
    def prepare_classification_data(self, n_groups: int = None) -> pd.DataFrame:
        """Prepare data for classification task."""
        
        log_message(f"Preparing classification data (n_groups={n_groups})...")
        
        if self.df is None:
            raise ValueError("Data not preprocessed yet")
        
        df = self.df.copy()
        
        # Get group counts
        group_counts = df['canonical_group'].value_counts()
        
        # Filter groups
        if n_groups and n_groups != 'all':
            top_groups = group_counts.head(n_groups).index.tolist()
            df = df[df['canonical_group'].isin(top_groups)]
        else:
            # Filter groups with minimum samples
            valid_groups = group_counts[group_counts >= self.config.MIN_SAMPLES_PER_GROUP].index
            df = df[df['canonical_group'].isin(valid_groups)]
        
        # Balance dataset
        balanced_dfs = []
        for group in df['canonical_group'].unique():
            group_df = df[df['canonical_group'] == group]
            if len(group_df) > self.config.MAX_SAMPLES_PER_GROUP:
                group_df = group_df.sample(n=self.config.MAX_SAMPLES_PER_GROUP, 
                                           random_state=self.config.SEED)
            balanced_dfs.append(group_df)
        
        result_df = pd.concat(balanced_dfs, ignore_index=True)
        
        log_message(f"Classification data: {len(result_df)} samples, "
                   f"{result_df['canonical_group'].nunique()} groups")
        
        # Save checkpoint
        save_checkpoint(result_df, f'classification_data_{n_groups}', self.config)
        
        return result_df


# ======================================================================
# DATA ANALYSIS
# ======================================================================

class DataAnalyzer:
    """Comprehensive data analysis and visualization."""
    
    def __init__(self, df: pd.DataFrame, config: Config = Config):
        self.df = df
        self.config = config
        self.analysis_results = {}
    
    def run_full_analysis(self):
        """Run complete data analysis."""
        
        log_message("="*60)
        log_message("📊 COMPREHENSIVE DATA ANALYSIS")
        log_message("="*60)
        
        self.basic_statistics()
        self.text_analysis()
        self.group_analysis()
        self.temporal_analysis()
        self.entity_coverage_analysis()
        
        self.save_analysis_report()
        
        return self.analysis_results
    
    def basic_statistics(self):
        """Calculate basic statistics."""
        
        log_message("\n📈 Basic Statistics:")
        
        stats = {
            'total_records': len(self.df),
            'unique_groups': self.df['canonical_group'].nunique() if 'canonical_group' in self.df.columns else 0,
        }
        
        if 'clean_text' in self.df.columns:
            text_lengths = self.df['clean_text'].str.len()
            stats['avg_text_length'] = float(text_lengths.mean())
            stats['min_text_length'] = int(text_lengths.min())
            stats['max_text_length'] = int(text_lengths.max())
            stats['median_text_length'] = float(text_lengths.median())
        
        if 'iyear' in self.df.columns:
            stats['year_range'] = [int(self.df['iyear'].min()), int(self.df['iyear'].max())]
        
        self.analysis_results['basic_stats'] = stats
        
        for key, value in stats.items():
            log_message(f"   {key}: {value}")
    
    def text_analysis(self):
        """Analyze text content with visualizations."""
        
        log_message("\n📝 Text Analysis:")
        
        if 'clean_text' not in self.df.columns:
            return
        
        text_lengths = self.df['clean_text'].str.len()
        word_counts = self.df['clean_text'].str.split().str.len()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Text length histogram
        ax1 = axes[0, 0]
        ax1.hist(text_lengths, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        ax1.axvline(text_lengths.mean(), color='red', linestyle='--', 
                   label=f'Mean: {text_lengths.mean():.0f}')
        ax1.axvline(text_lengths.median(), color='orange', linestyle='--',
                   label=f'Median: {text_lengths.median():.0f}')
        ax1.set_xlabel('Text Length (characters)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Text Lengths')
        ax1.legend()
        
        # Word count histogram
        ax2 = axes[0, 1]
        ax2.hist(word_counts, bins=50, color='forestgreen', edgecolor='white', alpha=0.7)
        ax2.axvline(word_counts.mean(), color='red', linestyle='--',
                   label=f'Mean: {word_counts.mean():.0f}')
        ax2.set_xlabel('Word Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Word Counts')
        ax2.legend()
        
        # Text source analysis
        ax3 = axes[1, 0]
        source_lengths = {}
        for col in ['summary', 'motive', 'addnotes']:
            if col in self.df.columns:
                lengths = self.df[col].dropna().astype(str).str.len()
                if len(lengths) > 0:
                    source_lengths[col] = lengths.mean()
        
        if source_lengths:
            ax3.bar(source_lengths.keys(), source_lengths.values(), 
                   color=['steelblue', 'forestgreen', 'coral'])
            ax3.set_ylabel('Average Length (chars)')
            ax3.set_title('Average Text Length by Source Column')
        
        # Box plot
        ax4 = axes[1, 1]
        ax4.boxplot([text_lengths, word_counts], labels=['Chars', 'Words'])
        ax4.set_ylabel('Count')
        ax4.set_title('Text Length Box Plot')
        
        plt.tight_layout()
        save_figure(fig, 'text_analysis', self.config)
        
        self.analysis_results['text_analysis'] = {
            'mean_length': float(text_lengths.mean()),
            'median_length': float(text_lengths.median()),
            'mean_words': float(word_counts.mean())
        }
        
        log_message(f"   Mean text length: {text_lengths.mean():.0f} chars")
        log_message(f"   Mean word count: {word_counts.mean():.0f} words")
    
    def group_analysis(self):
        """Analyze group distribution."""
        
        log_message("\n👥 Group Analysis:")
        
        if 'canonical_group' not in self.df.columns:
            return
        
        group_counts = self.df['canonical_group'].value_counts()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top 20 groups
        ax1 = axes[0, 0]
        top_20 = group_counts.head(20)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_20)))
        bars = ax1.barh(range(len(top_20)), top_20.values, color=colors)
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels(top_20.index, fontsize=9)
        ax1.set_xlabel('Number of Records')
        ax1.set_title('Top 20 Terrorism Groups')
        ax1.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars, top_20.values)):
            ax1.text(val + 50, i, str(val), va='center', fontsize=8)
        
        # Group size distribution (log scale)
        ax2 = axes[0, 1]
        ax2.hist(group_counts.values, bins=50, color='coral', edgecolor='white', alpha=0.7)
        ax2.set_xlabel('Number of Records per Group')
        ax2.set_ylabel('Number of Groups')
        ax2.set_title('Distribution of Records per Group')
        ax2.set_yscale('log')
        
        # Cumulative distribution
        ax3 = axes[1, 0]
        sorted_counts = np.sort(group_counts.values)[::-1]
        cumsum = np.cumsum(sorted_counts) / sorted_counts.sum() * 100
        ax3.plot(range(1, len(cumsum) + 1), cumsum, color='steelblue', linewidth=2)
        ax3.axhline(80, color='red', linestyle='--', alpha=0.7, label='80%')
        ax3.axhline(90, color='orange', linestyle='--', alpha=0.7, label='90%')
        ax3.set_xlabel('Number of Groups')
        ax3.set_ylabel('Cumulative % of Records')
        ax3.set_title('Cumulative Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Class imbalance visualization
        ax4 = axes[1, 1]
        size_categories = {
            '1-10': len(group_counts[group_counts <= 10]),
            '11-50': len(group_counts[(group_counts > 10) & (group_counts <= 50)]),
            '51-100': len(group_counts[(group_counts > 50) & (group_counts <= 100)]),
            '101-500': len(group_counts[(group_counts > 100) & (group_counts <= 500)]),
            '>500': len(group_counts[group_counts > 500])
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        ax4.pie(size_categories.values(), labels=size_categories.keys(),
               autopct='%1.1f%%', colors=colors)
        ax4.set_title('Groups by Record Count')
        
        plt.tight_layout()
        save_figure(fig, 'group_analysis', self.config)
        
        # Calculate imbalance ratio
        imbalance_ratio = group_counts.max() / group_counts.min() if group_counts.min() > 0 else float('inf')
        
        self.analysis_results['group_analysis'] = {
            'total_groups': len(group_counts),
            'top_group': group_counts.index[0],
            'top_count': int(group_counts.iloc[0]),
            'imbalance_ratio': float(imbalance_ratio),
            'size_categories': size_categories
        }
        
        log_message(f"   Total unique groups: {len(group_counts)}")
        log_message(f"   Top group: {group_counts.index[0]} ({group_counts.iloc[0]} records)")
        log_message(f"   Class imbalance ratio: {imbalance_ratio:.2f}")
    
    def temporal_analysis(self):
        """Analyze temporal patterns."""
        
        log_message("\n📅 Temporal Analysis:")
        
        if 'iyear' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Attacks over years
        ax1 = axes[0]
        yearly_counts = self.df['iyear'].value_counts().sort_index()
        ax1.plot(yearly_counts.index, yearly_counts.values, color='steelblue', 
                linewidth=2, marker='o', markersize=3)
        ax1.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Records')
        ax1.set_title('Records Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Top 5 groups over time
        ax2 = axes[1]
        if 'canonical_group' in self.df.columns:
            top_5 = self.df['canonical_group'].value_counts().head(5).index
            for group in top_5:
                group_yearly = self.df[self.df['canonical_group'] == group]['iyear'].value_counts().sort_index()
                ax2.plot(group_yearly.index, group_yearly.values, label=group[:20], linewidth=1.5)
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Number of Records')
            ax2.set_title('Top 5 Groups Over Time')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, 'temporal_analysis', self.config)
        
        self.analysis_results['temporal'] = {
            'year_range': [int(self.df['iyear'].min()), int(self.df['iyear'].max())],
            'peak_year': int(yearly_counts.idxmax())
        }
        
        log_message(f"   Year range: {self.df['iyear'].min()} - {self.df['iyear'].max()}")
        log_message(f"   Peak year: {yearly_counts.idxmax()} ({yearly_counts.max()} records)")
    
    def entity_coverage_analysis(self):
        """Analyze entity coverage in text."""
        
        log_message("\n🏷️ Entity Coverage Analysis:")
        
        if 'clean_text' not in self.df.columns:
            return
        
        coverage = {
            'terror_group': 0,
            'location': 0,
            'weapon': 0,
            'attack_type': 0
        }
        
        sample_size = min(5000, len(self.df))  # Analyze sample for speed
        sample_df = self.df.sample(n=sample_size, random_state=42)
        
        for _, row in tqdm(sample_df.iterrows(), total=sample_size, desc="Analyzing coverage"):
            text_lower = row['clean_text'].lower()
            
            # Terror group
            canonical = row.get('canonical_group', '')
            if canonical in TERRORISM_GAZETTEER:
                for alias in TERRORISM_GAZETTEER[canonical]:
                    if alias.lower() in text_lower:
                        coverage['terror_group'] += 1
                        break
            
            # Location
            for col in ['country_txt', 'city']:
                if col in row.index and pd.notna(row[col]):
                    if str(row[col]).lower() in text_lower:
                        coverage['location'] += 1
                        break
            
            # Weapon
            for alias in WEAPON_FLAT.keys():
                if alias in text_lower:
                    coverage['weapon'] += 1
                    break
            
            # Attack type
            for alias in ATTACK_TYPE_FLAT.keys():
                if alias in text_lower:
                    coverage['attack_type'] += 1
                    break
        
        # Convert to percentages
        coverage_pct = {k: v / sample_size * 100 for k, v in coverage.items()}
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax.bar(coverage_pct.keys(), coverage_pct.values(), color=colors, edgecolor='white')
        ax.set_ylabel('Percentage of Records (%)')
        ax.set_title('Entity Coverage in Text')
        ax.set_ylim(0, 100)
        
        for bar, pct in zip(bars, coverage_pct.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{pct:.1f}%', ha='center', fontsize=11)
        
        plt.tight_layout()
        save_figure(fig, 'entity_coverage', self.config)
        
        self.analysis_results['entity_coverage'] = coverage_pct
        
        for entity, pct in coverage_pct.items():
            log_message(f"   {entity}: {pct:.1f}% coverage")
    
    def save_analysis_report(self):
        """Save analysis report to JSON."""
        
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(i) for i in obj]
            return obj
        
        report = convert_types(self.analysis_results)
        
        report_path = self.config.RESULTS_DIR / "data_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        log_message(f"📋 Analysis report saved: {report_path}")


# ======================================================================
# MAIN EXECUTION
# ======================================================================

def main():
    """Main execution function."""
    
    print("="*70)
    print("🚀 TerrorismNER & Classification Research Project")
    print("   Data Processing & Analysis (CORRECTED VERSION)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Setup
    Config.setup_directories()
    Config.set_seeds()
    
    # Check PyTorch
    try:
        import torch
        print(f"\n✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\n⚠️ PyTorch not installed")
    
    # Initialize processor
    processor = DataProcessor()
    
    # Find GTD file
    gtd_path = Config.GTD_PATH
    if not os.path.exists(gtd_path):
        print(f"\n❌ GTD file not found at: {gtd_path}")
        uploads_dir = "/mnt/user-data/uploads"
        if os.path.exists(uploads_dir):
            excel_files = [f for f in os.listdir(uploads_dir) if f.endswith(('.xlsx', '.xls'))]
            if excel_files:
                gtd_path = os.path.join(uploads_dir, excel_files[0])
                Config.GTD_PATH = gtd_path
                print(f"✅ Using: {gtd_path}")
            else:
                print("No Excel files found")
                return
    
    try:
        # Load and preprocess
        print("\n" + "="*60)
        print("📂 STEP 1: LOADING & PREPROCESSING")
        print("="*60)
        
        df = processor.load_gtd(gtd_path)
        df = processor.preprocess()
        
        # Analyze normalization
        processor.analyze_group_normalization()
        
        # Run analysis
        print("\n" + "="*60)
        print("📊 STEP 2: DATA ANALYSIS")
        print("="*60)
        
        analyzer = DataAnalyzer(df)
        results = analyzer.run_full_analysis()
        
        # Prepare NER data
        print("\n" + "="*60)
        print("🏷️ STEP 3: NER DATA PREPARATION")
        print("="*60)
        
        train_basic, val_basic, test_basic = processor.prepare_ner_data(extended=False)
        train_ext, val_ext, test_ext = processor.prepare_ner_data(extended=True)
        
        # Prepare classification data
        print("\n" + "="*60)
        print("📝 STEP 4: CLASSIFICATION DATA PREPARATION")
        print("="*60)
        
        for n_groups in Config.GROUP_SIZES:
            clf_df = processor.prepare_classification_data(n_groups)
        
        print("\n" + "="*70)
        print("✅ DATA PREPARATION COMPLETE")
        print("="*70)
        print(f"\n📁 All files saved to: {Config.BASE_DIR}")
        print(f"   - Checkpoints: {Config.CHECKPOINTS_DIR}")
        print(f"   - Figures: {Config.FIGURES_DIR}")
        print(f"   - Results: {Config.RESULTS_DIR}")
        
        print("\n📋 SUMMARY:")
        print(f"   Total records: {len(df)}")
        print(f"   Unique groups: {df['canonical_group'].nunique()}")
        print(f"   NER samples (basic): {len(train_basic) + len(val_basic) + len(test_basic)}")
        print(f"   NER samples (extended): {len(train_ext) + len(val_ext) + len(test_ext)}")
        
        clear_memory()
        
    except Exception as e:
        log_message(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
