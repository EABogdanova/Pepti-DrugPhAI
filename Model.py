import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, matthews_corrcoef
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class PeptideClassifier(nn.Module):
    def __init__(self, emb_dims, hidden_dim=256, n_heads=4, dropout=0.3, num_classes=3):
        super().__init__()
        self.keys = sorted(emb_dims.keys())


        self.projectors = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(emb_dims[key], hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for key in self.keys
        })

        # 2. Attention Fusion
        self.query_vector = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout)

        # 3. Classification Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, embeddings_dict):
        batch_size = next(iter(embeddings_dict.values())).size(0)
        device = next(iter(embeddings_dict.values())).device

        projected_views = []
        masks = []

        # --- MODALITY DROPOUT ---
        if self.training:
            p_drop_modality = 0.1
            keep_probs = torch.rand(len(self.keys), device=device)
            if keep_probs.max() < p_drop_modality:
                keep_probs[torch.randint(0, len(self.keys), (1,))] = 1.0

            drop_mask = (keep_probs < p_drop_modality)
        else:
            drop_mask = [False] * len(self.keys)


        for i, key in enumerate(self.keys):
            tensor = embeddings_dict[key]

            is_missing = (torch.sum(torch.abs(tensor), dim=1) == 0)

            if self.training and drop_mask[i]:
                tensor = torch.zeros_like(tensor)
                is_missing = torch.ones_like(is_missing)

            masks.append(is_missing)
            projected_views.append(self.projectors[key](tensor))

        stack = torch.stack(projected_views, dim=1)
        key_padding_mask = torch.stack(masks, dim=1)

        all_missing = key_padding_mask.all(dim=1)
        key_padding_mask[all_missing, 0] = False

        query = self.query_vector.repeat(batch_size, 1, 1)
        fused_emb, attn_weights = self.attention(query, stack, stack, key_padding_mask=key_padding_mask)
        fused_emb = fused_emb.squeeze(1)

        logits = self.head(fused_emb)
        return logits, attn_weights


# --- Using ---
dims = {'esm': 1280, 'mtr': 1024, 'chemberta': 768}
model = PeptideClassifier(dims)
data = {
    'esm': torch.randn(10, 1280),
    'mtr': torch.zeros(10, 1024),
    'chemberta': torch.randn(10, 768)
 }
pred, weights = model(data)

# --- Dataset Class ---

class MultimodalClassificationDataset(Dataset):

    def __init__(self, embeddings_dict, y_values, task='classification'):

        self.embeddings = embeddings_dict
        self.task = task
        if self.task == 'classification':

            # Apply thresholds: <4h (log 0.602), 4-24h (log 1.38), >24h
            self.y = torch.tensor(self._digitize_y(y_values), dtype=torch.long)
        else:
            self.y = torch.tensor(y_values, dtype=torch.float32)

    def _digitize_y(self, y):
        # 0 - Short, 1 - Moderate, 2 - Long
        return np.where(y < -0.3, 0, np.where(y < 1, 1, 2))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item_embs = {k: v[idx] for k, v in self.embeddings.items()}
        return item_embs,  self.y[idx]

# --- CSV Loading and Merging Function ---

def load_and_align_data(path_chemberta, path_esm, path_peptmtr, label_col='y'):
    """
    Reads 3 files, extracts embeddings and labels.
    Assumptions:
    1. All columns except label_col and 'ID' are embedding dimensions.
    2. The order of rows in all files is identical.
    """
    print(f"Loading data from:\n - {path_chemberta}\n - {path_esm}\n - {path_peptmtr}")
    df_chem = pd.read_csv(path_chemberta)
    df_esm = pd.read_csv(path_esm)
    df_mtr = pd.read_csv(path_peptmtr)

    # Check for length match

    assert len(df_chem) == len(df_esm) == len(df_mtr), "Error: Files have different numbers of rows!"

    # Extract labels (taking from the first file; checking match with the others is good practice)
    y = df_chem[label_col].values
    # Function to extract only numeric columns (embeddings)

    def get_feats(df):
        # Exclude labels and IDs, if they exist
        cols_to_drop = [c for c in df.columns if c in [label_col, 'Org_ind', 'ID', 'id', 'Sequence', 'Smiles', 'Canonical_smiles']]
        return df.drop(columns=cols_to_drop).values.astype(np.float32)

    embeddings = {
        'chemberta': get_feats(df_chem),
        'esm': get_feats(df_esm),
        'mtr': get_feats(df_mtr)

    }
    print(f"Loaded {len(y)} samples.")
    print(f"Dimensions: Chem={embeddings['chemberta'].shape}, ESM={embeddings['esm'].shape}, Mtr={embeddings['mtr'].shape}")
    return embeddings, y