"""Two-tower retrieval model for Suno RecSys.

This implements a two-tower architecture with 2-layer MLP trained with
an InfoNCE-style loss using simulated positive/negative pairs.

Design choices:
- Item tower input: fixed item feature vectors (e.g. CLAP audio embeddings,
  possibly concatenated with other numeric metadata) of size D_base.
- Item tower: 2-layer MLP (Linear -> ReLU -> Dropout -> Linear) to D_model, followed by L2 norm.
- User tower input: average of last-n interacted item feature vectors
  (same D_base as item tower input).
- User tower: the same structure as item tower (2-layer MLP + L2 norm), so
  user and item embeddings are aligned in R^{D_model}.
- Temperature scaling: Applied to InfoNCE logits for stable training.
- Learning rate scheduling: CosineAnnealingLR for better convergence.

Prompt embeddings are NOT used inside this two-tower model per design –
prompt-based exploration is handled by a separate prompt-only channel.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class TwoTowerConfig:
    """Configuration for two-tower model and training."""

    base_dim: int          # Dimension of raw item feature vectors (e.g. CLAP dim)
    model_dim: int = 256   # Shared dimension for user/item embeddings
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_negatives: int = 10
    num_epochs: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temperature: float = 0.1  # Temperature scaling for InfoNCE loss
    hidden_dim: int = 512     # Hidden dimension for MLP 
    dropout: float = 0.1       # Dropout rate for MLP


class TwoTowerModel(nn.Module):
    """Two-tower model with 2-layer MLP and L2-normalized outputs."""

    def __init__(self, config: TwoTowerConfig):
        super().__init__()
        self.config = config

        # Item tower: R^{D_base} -> R^{hidden_dim} -> R^{D_model}
        self.item_proj = nn.Sequential(
            nn.Linear(config.base_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim//2, config.model_dim),
        )

        # User tower: R^{D_base} -> R^{hidden_dim} -> R^{D_model}
        self.user_proj = nn.Sequential(
            nn.Linear(config.base_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim//2, config.model_dim),
        )

    def encode_items(self, item_feats: torch.Tensor) -> torch.Tensor:
        """Encode item features into normalized embeddings."""
        z = self.item_proj(item_feats)
        z = nn.functional.normalize(z, dim=-1)
        return z

    def encode_users(self, user_feats: torch.Tensor) -> torch.Tensor:
        """Encode user features into normalized embeddings."""
        z = self.user_proj(user_feats)
        z = nn.functional.normalize(z, dim=-1)
        return z

    def forward(
        self,
        user_feats: torch.Tensor,
        pos_item_feats: torch.Tensor,
        neg_item_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for a batch.

        Args:
            user_feats: (B, D_base)
            pos_item_feats: (B, D_base)
            neg_item_feats: (B, K, D_base)

        Returns:
            Scalar loss tensor.
        """
        u = self.encode_users(user_feats)          # (B, D_model)
        i_pos = self.encode_items(pos_item_feats)  # (B, D_model)

        B, K, D_base = neg_item_feats.shape
        neg_item_feats = neg_item_feats.view(B * K, D_base)
        i_neg = self.encode_items(neg_item_feats).view(B, K, -1)  # (B, K, D_model)

        # Positive logits: (B, 1)
        pos_logits = (u * i_pos).sum(dim=-1, keepdim=True)  # (B, 1)

        # Negative logits: (B, K)
        neg_logits = torch.bmm(i_neg, u.unsqueeze(-1)).squeeze(-1)  # (B, K)

        # Apply temperature scaling
        temperature = self.config.temperature
        pos_logits = pos_logits / temperature
        neg_logits = neg_logits / temperature

        logits = torch.cat([pos_logits, neg_logits], dim=-1)  # (B, 1+K)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # positives at index 0

        loss = nn.functional.cross_entropy(logits, labels)
        return loss


class TwoTowerDataset(Dataset):
    """
    Dataset that yields (user_feat, pos_item_feat, neg_item_feats) triples.

    - user_feat: average of positive item feature vectors for that user (D_base).
    - pos_item_feat: feature vector of a positive item (D_base).
    - neg_item_feats: K negative item feature vectors (K, D_base) sampled from
      the user's negative items.
    """

    def __init__(
        self,
        events_df: pd.DataFrame,
        songs_df: pd.DataFrame,
        item_features: np.ndarray,
        song_id_to_idx: Dict[str, int],
        num_negatives: int = 10,
    ):
        """
        Args:
            events_df: DataFrame with columns [user_id, song_id, label].
            songs_df: DataFrame with song metadata (must contain song_id).
            item_features: np.ndarray of shape (num_items, D_base), aligned with song_id_to_idx.
            song_id_to_idx: mapping from song_id to row index in item_features.
            num_negatives: number of negatives per positive sample.
        """
        self.num_negatives = num_negatives
        self.item_features = item_features.astype(np.float32)
        self.song_id_to_idx = song_id_to_idx

        # Build per-user positives / negatives
        # Use tqdm if available for large datasets
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        grouped = events_df.groupby("user_id")
        self.user_ids: List[str] = []
        self.user_pos_items: Dict[str, List[int]] = {}
        self.user_neg_items: Dict[str, List[int]] = {}

        iterator = tqdm(grouped, desc="Building user-item mappings") if use_tqdm else grouped
        
        for uid, group in iterator:
            pos_song_ids = group.loc[group["label"] == 1, "song_id"].tolist()
            neg_song_ids = group.loc[group["label"] == 0, "song_id"].tolist()

            pos_indices = [song_id_to_idx[sid] for sid in pos_song_ids if sid in song_id_to_idx]
            neg_indices = [song_id_to_idx[sid] for sid in neg_song_ids if sid in song_id_to_idx]

            if not pos_indices or not neg_indices:
                continue

            self.user_ids.append(uid)
            self.user_pos_items[uid] = pos_indices
            self.user_neg_items[uid] = neg_indices

        # Build a flat list of (user_id, pos_idx) pairs for iteration
        self.samples: List[Tuple[str, int]] = []
        for uid in self.user_ids:
            for pos_idx in self.user_pos_items[uid]:
                self.samples.append((uid, pos_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        uid, pos_idx = self.samples[idx]
        pos_list = self.user_pos_items[uid]
        neg_list = self.user_neg_items[uid]

        # user feature: average of all positive items for this user
        user_feat = self.item_features[pos_list].mean(axis=0)

        pos_item_feat = self.item_features[pos_idx]

        # sample negatives without replacement
        if len(neg_list) <= self.num_negatives:
            neg_indices = neg_list
        else:
            neg_indices = np.random.choice(neg_list, size=self.num_negatives, replace=False)

        neg_item_feats = self.item_features[neg_indices]

        return (
            torch.from_numpy(user_feat),
            torch.from_numpy(pos_item_feat),
            torch.from_numpy(neg_item_feats),
        )


def build_item_feature_matrix(
    songs_df: pd.DataFrame,
    song_id_to_idx: Dict[str, int],
    clap_embeddings: Dict[str, np.ndarray],
    target_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Build item feature matrix aligned with song_id_to_idx using CLAP embeddings.

    If a song_id is missing in clap_embeddings, a small random vector is used.
    All vectors are truncated/padded to the same base_dim, then optionally
    projected (via PCA-style truncation) to target_dim if provided.
    """
    num_items = len(song_id_to_idx)

    # Determine base_dim from first CLAP vector
    first_emb = next(iter(clap_embeddings.values()))
    base_dim = len(first_emb)

    feats = np.zeros((num_items, base_dim), dtype=np.float32)

    # Build reverse mapping from idx to song_id
    idx_to_song_id = {idx: sid for sid, idx in song_id_to_idx.items()}

    for idx in range(num_items):
        song_id = idx_to_song_id[idx]
        emb = clap_embeddings.get(song_id)
        if emb is None:
            # Fallback: small random vector
            emb = np.random.normal(scale=0.01, size=base_dim)
        else:
            emb = np.asarray(emb, dtype=np.float32)
            if emb.shape[0] != base_dim:
                # Truncate or pad
                if emb.shape[0] > base_dim:
                    emb = emb[:base_dim]
                else:
                    padded = np.zeros(base_dim, dtype=np.float32)
                    padded[: emb.shape[0]] = emb
                    emb = padded
        feats[idx] = emb

    # Optionally reduce to target_dim by simple truncation
    if target_dim is not None and target_dim < base_dim:
        feats = feats[:, :target_dim]

    # L2 normalize
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats = feats / norms

    return feats


def train_two_tower(
    config: TwoTowerConfig,
    events_df: pd.DataFrame,
    songs_df: pd.DataFrame,
    song_id_to_idx: Dict[str, int],
    item_features: np.ndarray,
    checkpoint_path: Optional[str] = None,
    save_checkpoint_path: Optional[str] = None,
    save_every_n_epochs: int = 1,
    use_wandb: bool = False,
    wandb_project: str = "sunorecsys-two-tower",
    wandb_run_name: Optional[str] = None,
    final_checkpoint_path: Optional[str] = None,
) -> Tuple[TwoTowerModel, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Train a two-tower model with InfoNCE-style loss on simulated events.

    Args:
        config: Training configuration
        events_df: DataFrame with positive/negative events
        songs_df: DataFrame with song metadata
        song_id_to_idx: Mapping from song_id to index
        item_features: Item feature matrix
        checkpoint_path: Optional path to checkpoint to resume from
        save_checkpoint_path: Optional path to save checkpoints during training
        save_every_n_epochs: Save checkpoint every N epochs (default: 1, save every epoch)
        use_wandb: Whether to use wandb for logging
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name (optional)
        final_checkpoint_path: Optional path to save final checkpoint with optimizer/scheduler state

    Returns:
        Tuple of (Trained TwoTowerModel, Optimizer, Scheduler).
    """
    dataset = TwoTowerDataset(
        events_df=events_df,
        songs_df=songs_df,
        item_features=item_features,
        song_id_to_idx=song_id_to_idx,
        num_negatives=config.num_negatives,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    base_dim = item_features.shape[1]
    if base_dim != config.base_dim:
        raise ValueError(f"config.base_dim={config.base_dim}, but item_features.shape[1]={base_dim}")

    model = TwoTowerModel(config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.lr * 0.01,  # Minimum LR is 1% of initial LR
    )
    
    # Initialize wandb if enabled
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    **config.__dict__,
                    'num_samples': len(dataset),
                    'num_batches': len(dataloader),
                },
                reinit=True,
            )
            # Log model architecture
            wandb.watch(model, log='all', log_freq=100)
        except ImportError:
            print("⚠️  wandb not installed. Install with: pip install wandb")
            use_wandb = False

    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"📂 Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        
        # Load model state
        model.load_state_dict(checkpoint['state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Resume from epoch
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"   Resuming from epoch {start_epoch}/{config.num_epochs}")
        else:
            print(f"   ⚠️  No epoch info in checkpoint, starting from epoch 0")
        
        # Verify config matches (warn if different)
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            if isinstance(checkpoint_config, dict):
                for key in ['base_dim', 'model_dim']:
                    if key in checkpoint_config and checkpoint_config[key] != getattr(config, key):
                        print(f"   ⚠️  Config mismatch: {key} = {checkpoint_config[key]} (checkpoint) vs {getattr(config, key)} (current)")
        
        print(f"✅ Checkpoint loaded successfully")
    else:
        if checkpoint_path:
            print(f"⚠️  Checkpoint file not found: {checkpoint_path}, starting from scratch")

    # Add progress bar if tqdm available
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    model.train()
    for epoch in range(start_epoch, config.num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}") if use_tqdm else dataloader
        
        for batch_idx, (user_feat, pos_feat, neg_feats) in enumerate(iterator):
            user_feat = user_feat.to(config.device)
            pos_feat = pos_feat.to(config.device)
            neg_feats = neg_feats.to(config.device)

            optimizer.zero_grad()
            loss = model(user_feat, pos_feat, neg_feats)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
            if use_tqdm:
                iterator.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to wandb (every 10 batches to avoid too much logging)
            if use_wandb and batch_idx % 10 == 0:
                try:
                    import wandb
                    wandb.log({
                        'batch_loss': loss.item(),
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'epoch': epoch + 1,
                        'batch': batch_idx,
                    })
                except ImportError:
                    pass

        avg_loss = total_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[TwoTower] Epoch {epoch+1}/{config.num_epochs} - loss: {avg_loss:.4f} - lr: {current_lr:.6f}")
        
        # Log epoch metrics to wandb
        if use_wandb:
            try:
                import wandb
                wandb.log({
                    'epoch_loss': avg_loss,
                    'learning_rate': current_lr,
                    'epoch': epoch + 1,
                })
            except ImportError:
                pass
        
        # Step the learning rate scheduler
        scheduler.step()

        # Save checkpoint if enabled
        if save_checkpoint_path and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_data = {
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config.__dict__,
                'epoch': epoch,
                'loss': avg_loss,
                'song_id_to_idx': song_id_to_idx,
            }
            
            # Save with epoch number in filename
            checkpoint_file = Path(save_checkpoint_path)
            if checkpoint_file.suffix == '.pt':
                # Replace .pt with _epoch_{epoch}.pt
                checkpoint_path_epoch = checkpoint_file.parent / f"{checkpoint_file.stem}_epoch_{epoch+1}.pt"
            else:
                checkpoint_path_epoch = checkpoint_file.parent / f"{checkpoint_file.name}_epoch_{epoch+1}.pt"
            
            torch.save(checkpoint_data, str(checkpoint_path_epoch))
            print(f"   💾 Saved checkpoint to {checkpoint_path_epoch}")
    
    # Save final checkpoint with optimizer and scheduler state if requested
    if final_checkpoint_path:
        final_epoch = config.num_epochs - 1
        final_checkpoint_data = {
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config.__dict__,
            'epoch': final_epoch,
            'loss': avg_loss if 'avg_loss' in locals() else None,
            'song_id_to_idx': song_id_to_idx,
        }
        Path(final_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_checkpoint_data, final_checkpoint_path)
        print(f"   💾 Saved final checkpoint (with optimizer/scheduler) to {final_checkpoint_path}")
    
    # Finish wandb run
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass

    return model, optimizer, scheduler



