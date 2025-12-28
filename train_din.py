"""Training script for the Deep Interest Network (DIN) CTR prediction model.

Example usage:

    # Basic training
    python train_din.py \\
        --songs sunorecsys/datasets/curl/all_playlist_songs.json \\
        --clap-embeddings runtime_data/clap_embeddings.json \\
        --output model_checkpoints/din_ranker.pt

    # Training with checkpoint saving
    python train_din.py \\
        --songs sunorecsys/datasets/curl/all_playlist_songs.json \\
        --clap-embeddings runtime_data/clap_embeddings.json \\
        --output model_checkpoints/din_ranker.pt \\
        --epochs 10 \\
        --batch-size 128 \\
        --max-history 50 \\
        --save-checkpoints model_checkpoints/din_checkpoint.pt \\
        --save-every-n-epochs 2

    # Resume from checkpoint
    python train_din.py \\
        --songs sunorecsys/datasets/curl/all_playlist_songs.json \\
        --clap-embeddings runtime_data/clap_embeddings.json \\
        --output model_checkpoints/din_ranker.pt \\
        --epochs 10 \\
        --resume-from model_checkpoints/din_checkpoint_epoch_5.pt

This will:
1. Load songs from all_playlist_songs.json (3881 songs by default).
2. Build simulated user-item interactions (positive/negative pairs).
3. Load CLAP embeddings as track features.
4. Train DIN model with attention-based aggregation for CTR prediction.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sunorecsys.datasets.preprocess import SongDataProcessor
from sunorecsys.datasets.simulate_interactions import load_songs_from_aggregated_file
from sunorecsys.datasets.simulate_interactions import get_user_item_matrix
from sunorecsys.recommenders.din_ranker import DINModel


def load_clap_embeddings(path: str) -> dict:
    """Load CLAP embeddings from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    # Ensure numpy arrays
    return {sid: np.asarray(vec, dtype=np.float32) for sid, vec in data.items()}


class DINDataset(Dataset):
    """Dataset for DIN training: (user_history, candidate_track, label)."""
    
    def __init__(
        self,
        events_df: pd.DataFrame,
        user_history: Dict[str, List[str]],
        clap_embeddings: Dict[str, np.ndarray],
        max_history: int = 50,
        embedding_dim: int = 512,
    ):
        """
        Initialize DIN dataset.
        
        Args:
            events_df: DataFrame with columns [user_id, song_id, label]
            user_history: Dict mapping user_id to list of historical song_ids
            clap_embeddings: Dict mapping song_id to CLAP embedding vector
            max_history: Maximum number of historical tracks to use
            embedding_dim: Dimension of CLAP embeddings
        """
        self.max_history = max_history
        self.embedding_dim = embedding_dim
        self.clap_embeddings = clap_embeddings
        
        # Build samples: (user_id, candidate_song_id, label, historical_song_ids)
        self.samples = []
        
        for _, row in events_df.iterrows():
            user_id = row['user_id']
            candidate_song_id = row['song_id']
            label = int(row['label'])
            
            # Get user's historical tracks (excluding the candidate if it's positive)
            historical_song_ids = user_history.get(user_id, [])
            if label == 1 and candidate_song_id in historical_song_ids:
                # For positive samples, exclude the candidate from history
                historical_song_ids = [sid for sid in historical_song_ids if sid != candidate_song_id]
            
            # Only include samples where both candidate and history have embeddings
            if candidate_song_id in clap_embeddings:
                # Filter history to only include tracks with embeddings
                valid_history = [sid for sid in historical_song_ids if sid in clap_embeddings]
                if len(valid_history) > 0 or label == 0:  # Allow negative samples with no history
                    self.samples.append({
                        'user_id': user_id,
                        'candidate_song_id': candidate_song_id,
                        'label': label,
                        'historical_song_ids': valid_history[:max_history]  # Truncate to max_history
                    })
        
        print(f"  ✅ Created {len(self.samples)} training samples")
        num_positives = sum(s['label'] for s in self.samples)
        num_negatives = len(self.samples) - num_positives
        print(f"     Positives: {num_positives}, Negatives: {num_negatives}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get candidate embedding
        candidate_emb = self.clap_embeddings[sample['candidate_song_id']]
        
        # Get historical embeddings
        historical_song_ids = sample['historical_song_ids']
        
        if len(historical_song_ids) == 0:
            # No history: use zero embedding (will be masked)
            historical_embs = np.zeros((1, self.embedding_dim), dtype=np.float32)
            mask = np.array([0], dtype=np.float32)  # Mask out the zero embedding
        else:
            historical_embs = np.array([
                self.clap_embeddings[sid] for sid in historical_song_ids
            ], dtype=np.float32)
            mask = np.ones(len(historical_song_ids), dtype=np.float32)
        
        # Pad or truncate to max_history
        if len(historical_embs) < self.max_history:
            padding = np.zeros((self.max_history - len(historical_embs), self.embedding_dim), dtype=np.float32)
            historical_embs = np.vstack([historical_embs, padding])
            mask_padding = np.zeros(self.max_history - len(mask), dtype=np.float32)
            mask = np.concatenate([mask, mask_padding])
        else:
            historical_embs = historical_embs[:self.max_history]
            mask = mask[:self.max_history]
        
        return {
            'historical_embs': torch.from_numpy(historical_embs).float(),
            'candidate_emb': torch.from_numpy(candidate_emb).float(),
            'mask': torch.from_numpy(mask).float(),
            'label': torch.tensor(sample['label'], dtype=torch.float32),
        }


def train_din(
    model: DINModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float = 0.001,
    device: str = 'cuda',
    checkpoint_path: Optional[str] = None,
    save_checkpoint_path: Optional[str] = None,
    save_every_n_epochs: int = 1,
    use_wandb: bool = False,
    wandb_project: str = "sunorecsys-din",
    wandb_run_name: Optional[str] = None,
) -> Tuple[DINModel, optim.Optimizer]:
    """Train DIN model."""
    model = model.to(device)
    model.train()
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  → Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        print(f"  ✅ Resumed from epoch {start_epoch}")
    
    # Initialize wandb if requested
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    'batch_size': train_loader.batch_size,
                    'max_history': train_loader.dataset.max_history,
                }
            )
        except ImportError:
            print("  ⚠️  wandb not installed, skipping logging")
            use_wandb = False
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            historical_embs = batch['historical_embs'].to(device)  # (batch, max_history, emb_dim)
            candidate_embs = batch['candidate_emb'].to(device)  # (batch, emb_dim)
            masks = batch['mask'].to(device)  # (batch, max_history)
            labels = batch['label'].to(device)  # (batch,)
            
            optimizer.zero_grad()
            
            # Forward pass (with mask)
            ctr_pred, attention_weights = model(historical_embs, candidate_embs, mask=masks)
            
            # Compute loss
            loss = criterion(ctr_pred, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            predictions = (ctr_pred > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_pbar:
                historical_embs = batch['historical_embs'].to(device)
                candidate_embs = batch['candidate_emb'].to(device)
                masks = batch['mask'].to(device)
                labels = batch['label'].to(device)
                
                ctr_pred, attention_weights = model(historical_embs, candidate_embs, mask=masks)
                loss = criterion(ctr_pred, labels)
                
                val_loss += loss.item()
                predictions = (ctr_pred > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{val_correct/val_total:.4f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Logging
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr'],
            })
        
        # Save checkpoint
        if save_checkpoint_path and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path_epoch = save_checkpoint_path.replace('.pt', f'_epoch_{epoch+1}.pt')
            Path(checkpoint_path_epoch).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
            }, checkpoint_path_epoch)
            print(f"  💾 Saved checkpoint: {checkpoint_path_epoch}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_checkpoint_path:
                best_path = save_checkpoint_path.replace('.pt', '_best.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'val_acc': val_acc,
                }, best_path)
                print(f"  🏆 Saved best model: {best_path} (val_loss={avg_val_loss:.4f})")
    
    if use_wandb:
        wandb.finish()
    
    return model, optimizer


def build_user_history_from_events(events_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build user history from events DataFrame (positive interactions only)."""
    user_history = {}
    
    # Only use positive interactions (label=1) for history
    positive_events = events_df[events_df['label'] == 1].copy()
    
    # Sort by user_id to group by user
    positive_events = positive_events.sort_values('user_id')
    
    for user_id, group in positive_events.groupby('user_id'):
        user_history[user_id] = group['song_id'].tolist()
    
    return user_history


def main():
    parser = argparse.ArgumentParser(description="Train DIN CTR prediction model")
    parser.add_argument(
        "--songs",
        default="sunorecsys/datasets/curl/all_playlist_songs.json",
        help="Path to songs JSON file (default: all_playlist_songs.json with 3881 songs)",
    )
    parser.add_argument(
        "--clap-embeddings",
        required=True,
        help="Path to JSON file with CLAP embeddings (song_id -> vector)",
    )
    parser.add_argument(
        "--output",
        default="model_checkpoints/din_ranker.pt",
        help="Output path for trained DIN model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=50,
        help="Maximum number of historical tracks to use",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs='+',
        default=[128, 64],
        help="Hidden dimensions for MLP (default: 128 64)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--save-checkpoints",
        type=str,
        default=None,
        help="Path to save checkpoints during training (e.g., model_checkpoints/din_checkpoint.pt). "
             "Checkpoints will be saved as {path}_epoch_{epoch}.pt",
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1, save every epoch)",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use wandb for training monitoring",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="sunorecsys-din",
        help="Wandb project name (default: sunorecsys-din)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name (optional, will auto-generate if not provided)",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=10,
        help="Number of negatives per positive sample (for event generation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load songs from JSON file
    print("\n" + "="*80)
    print("Loading songs...")
    print("="*80)
    songs_file = Path(args.songs)
    if songs_file.is_file() and songs_file.suffix == '.json':
        print(f"📂 Loading from JSON file: {args.songs}")
        songs_df = load_songs_from_aggregated_file(str(songs_file), max_songs=None)
    else:
        print(f"📂 Loading from processed directory: {args.songs}")
        processor = SongDataProcessor()
        songs_df = processor.load_processed(args.songs)
    
    print(f"✅ Loaded {len(songs_df)} songs")
    
    # 2. Build user-item matrix and events (simulated interactions with negatives)
    print("\n" + "="*80)
    print("Building simulated user-item interactions (with events)...")
    print("="*80)
    matrix_and_mappings = get_user_item_matrix(
        songs_df=songs_df,
        aggregated_file=None,
        playlists=None,
        use_simulation=True,
        interaction_rate=0.15,
        item_cold_start_rate=0.05,
        single_user_item_rate=0.15,
        random_seed=args.seed,
        max_songs=None,
        cache_dir="runtime_data/cache",
        use_cache=True,
        return_events=True,
        num_negatives_per_user=args.num_negatives * 2,
    )
    
    (
        user_item_matrix,
        user_id_to_idx,
        idx_to_user_id,
        song_id_to_idx,
        idx_to_song_id,
        events_df,
    ) = matrix_and_mappings
    
    num_positives = (events_df['label'] == 1).sum()
    num_negatives = (events_df['label'] == 0).sum()
    print(f"✅ Events: {len(events_df)} rows ({num_positives} positives, {num_negatives} negatives)")
    
    # 3. Build user history from positive events
    print("\n" + "="*80)
    print("Building user history from positive interactions...")
    print("="*80)
    user_history = build_user_history_from_events(events_df)
    print(f"✅ Built history for {len(user_history)} users")
    avg_history_len = np.mean([len(h) for h in user_history.values()])
    print(f"   Average history length: {avg_history_len:.1f} tracks")
    
    # 4. Load CLAP embeddings
    print("\n" + "="*80)
    print(f"Loading CLAP embeddings from {args.clap_embeddings}...")
    print("="*80)
    clap_embeddings = load_clap_embeddings(args.clap_embeddings)
    print(f"✅ Loaded {len(clap_embeddings)} CLAP embeddings")
    
    # Get embedding dimension
    first_emb = next(iter(clap_embeddings.values()))
    embedding_dim = first_emb.shape[0]
    print(f"   Embedding dimension: {embedding_dim}")
    
    # 5. Create train/val split
    print("\n" + "="*80)
    print("Creating train/validation split...")
    print("="*80)
    # Shuffle events
    events_df_shuffled = events_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val_size = int(len(events_df_shuffled) * args.val_split)
    train_events = events_df_shuffled.iloc[val_size:]
    val_events = events_df_shuffled.iloc[:val_size]
    print(f"✅ Train: {len(train_events)} samples, Val: {len(val_events)} samples")
    
    # 6. Create datasets
    print("\n" + "="*80)
    print("Creating datasets...")
    print("="*80)
    train_dataset = DINDataset(
        events_df=train_events,
        user_history=user_history,
        clap_embeddings=clap_embeddings,
        max_history=args.max_history,
        embedding_dim=embedding_dim,
    )
    val_dataset = DINDataset(
        events_df=val_events,
        user_history=user_history,
        clap_embeddings=clap_embeddings,
        max_history=args.max_history,
        embedding_dim=embedding_dim,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if device == 'cuda' else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False,
    )
    
    # 7. Initialize DIN model
    print("\n" + "="*80)
    print("Initializing DIN model...")
    print("="*80)
    model = DINModel(
        embedding_dim=embedding_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )
    print(f"✅ Model initialized")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Hidden dims: {args.hidden_dims}")
    print(f"   Dropout: {args.dropout}")
    print(f"   Max history: {args.max_history}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 8. Train model
    print("\n" + "="*80)
    print("Training DIN model...")
    print("="*80)
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Device: {device}")
    if args.resume_from:
        print(f"   Resuming from: {args.resume_from}")
    if args.save_checkpoints:
        print(f"   Saving checkpoints to: {args.save_checkpoints} (every {args.save_every_n_epochs} epochs)")
    if args.use_wandb:
        print(f"   Wandb: project={args.wandb_project}, run={args.wandb_run_name or 'auto'}")
    
    model, optimizer = train_din(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_path=args.resume_from,
        save_checkpoint_path=args.save_checkpoints,
        save_every_n_epochs=args.save_every_n_epochs,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    
    # 9. Save final model
    print("\n" + "="*80)
    print("Saving final model...")
    print("="*80)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in format expected by DINRanker
    save_obj = {
        'model_state_dict': model.state_dict(),
        'embedding_dim': embedding_dim,
        'model_trained': True,  # Mark as trained
    }
    torch.save(save_obj, str(output_path))
    
    print(f"✅ DIN model saved to {output_path}")
    print(f"\n🎉 Training complete!")


if __name__ == "__main__":
    main()

