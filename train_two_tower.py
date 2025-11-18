"""Training script for the two-tower retrieval model using simulated interactions.

Example usage:

    # Basic training
    python train_two_tower.py \\
        --songs sunorecsys/data/curl/all_playlist_songs.json \\
        --clap-embeddings data/clap_embeddings.json \\
        --output models/two_tower.pt

    # Training with checkpoint saving
    python train_two_tower.py \\
        --songs sunorecsys/data/curl/all_playlist_songs.json \\
        --clap-embeddings data/clap_embeddings.json \\
        --output models/two_tower.pt \\
        --epochs 20 \\
        --save-checkpoints models/two_tower_checkpoint.pt \\
        --save-every-n-epochs 1

    # Resume from checkpoint
    python train_two_tower.py \\
        --songs sunorecsys/data/curl/all_playlist_songs.json \\
        --clap-embeddings data/clap_embeddings.json \\
        --output models/two_tower.pt \\
        --epochs 20 \\
        --resume-from models/two_tower_checkpoint_epoch_10.pt

This will:
1. Load songs from all_playlist_songs.json (3881 songs by default).
2. Build / load a simulated user-item matrix and event table (positive/negative pairs).
3. Load CLAP embeddings as item features.
4. Train a simple two-tower model with InfoNCE loss.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sunorecsys.data.preprocess import SongDataProcessor
from sunorecsys.data.simulate_interactions import load_songs_from_aggregated_file
from sunorecsys.data.simulate_interactions import get_user_item_matrix
from sunorecsys.recommenders.two_tower import (
    TwoTowerConfig,
    build_item_feature_matrix,
    train_two_tower,
)


def load_clap_embeddings(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    # Ensure numpy arrays
    return {sid: np.asarray(vec, dtype=np.float32) for sid, vec in data.items()}


def main():
    parser = argparse.ArgumentParser(description="Train two-tower retrieval model")
    parser.add_argument(
        "--songs",
        default="sunorecsys/data/curl/all_playlist_songs.json",
        help="Path to songs JSON file (default: all_playlist_songs.json with 3881 songs)",
    )
    parser.add_argument(
        "--clap-embeddings",
        required=True,
        help="Path to JSON file with CLAP embeddings (song_id -> vector)",
    )
    parser.add_argument(
        "--output",
        default="models/two_tower.pt",
        help="Output path for trained two-tower model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=10,
        help="Number of negatives per positive sample",
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
        help="Path to save checkpoints during training (e.g., models/two_tower_checkpoint.pt). "
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
        default="sunorecsys-two-tower",
        help="Wandb project name (default: sunorecsys-two-tower)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name (optional, will auto-generate if not provided)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature scaling for InfoNCE loss (default: 0.1)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for MLP (default: 512)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for MLP (default: 0.1)",
    )

    args = parser.parse_args()

    # 1. Load songs from JSON file
    print("Loading songs...")
    songs_file = Path(args.songs)
    if songs_file.is_file() and songs_file.suffix == '.json':
        # Load from JSON file (all_playlist_songs.json format)
        print(f"📂 Loading from JSON file: {args.songs}")
        songs_df = load_songs_from_aggregated_file(str(songs_file), max_songs=None)
    else:
        # Fallback: try as processed directory
        print(f"📂 Loading from processed directory: {args.songs}")
        processor = SongDataProcessor()
        songs_df = processor.load_processed(args.songs)
    
    print(f"✅ Loaded {len(songs_df)} songs")
    print(f"   Unique users: {songs_df['user_id'].nunique() if 'user_id' in songs_df.columns else 'N/A'}")

    # 2. Build user-item matrix and events (simulated interactions with negatives)
    print("Building simulated user-item interactions (with events)...")
    matrix_and_mappings = get_user_item_matrix(
        songs_df=songs_df,
        aggregated_file=None,
        playlists=None,
        use_simulation=True,
        interaction_rate=0.15,
        item_cold_start_rate=0.05,
        single_user_item_rate=0.15,
        random_seed=42,
        max_songs=None,
        cache_dir="data/cache",
        use_cache=True,
        return_events=True,
        num_negatives_per_user=args.num_negatives * 2,  # a few more negatives for sampling
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
    print(f"   Estimated training samples: ~{num_positives} (each with {args.num_negatives} negatives)")

    # 3. Load CLAP embeddings and build item feature matrix
    print(f"\nLoading CLAP embeddings from {args.clap_embeddings} ...")
    clap_embeddings = load_clap_embeddings(args.clap_embeddings)
    print(f"✅ Loaded {len(clap_embeddings)} CLAP embeddings")

    # Use CLAP dimensionality as base_dim
    first_emb = next(iter(clap_embeddings.values()))
    base_dim = first_emb.shape[0]
    print(f"   CLAP base_dim = {base_dim}")

    print("Building item feature matrix...")
    item_features = build_item_feature_matrix(
        songs_df=songs_df,
        song_id_to_idx=song_id_to_idx,
        clap_embeddings=clap_embeddings,
        target_dim=None,  # keep original CLAP dim
    )
    print(f"✅ Item feature matrix: {item_features.shape} (memory: ~{item_features.nbytes / 1024 / 1024:.1f} MB)")

    # 4. Train two-tower model
    config = TwoTowerConfig(
        base_dim=item_features.shape[1],
        model_dim=256,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        num_epochs=args.epochs,
        temperature=args.temperature,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    print(f"\nTraining two-tower model...")
    print(f"   Config: model_dim={config.model_dim}, hidden_dim={config.hidden_dim}, "
          f"batch_size={config.batch_size}, num_negatives={config.num_negatives}, "
          f"epochs={config.num_epochs}, temperature={config.temperature}, dropout={config.dropout}")
    print(f"   Device: {config.device}")
    if args.resume_from:
        print(f"   Resuming from checkpoint: {args.resume_from}")
    if args.save_checkpoints:
        print(f"   Saving checkpoints to: {args.save_checkpoints} (every {args.save_every_n_epochs} epochs)")
    if args.use_wandb:
        print(f"   Wandb: project={args.wandb_project}, run={args.wandb_run_name or 'auto'}")
    
    model, optimizer, scheduler = train_two_tower(
        config=config,
        events_df=events_df,
        songs_df=songs_df,
        song_id_to_idx=song_id_to_idx,
        item_features=item_features,
        checkpoint_path=args.resume_from,
        save_checkpoint_path=args.save_checkpoints,
        save_every_n_epochs=args.save_every_n_epochs,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        final_checkpoint_path=args.output,  # Save final checkpoint with optimizer/scheduler
    )

    # 5. Save final model with optimizer and scheduler state (for resuming)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch = __import__("torch")
    save_obj = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.__dict__,
        "song_id_to_idx": song_id_to_idx,
        "epoch": config.num_epochs - 1,  # Final epoch (0-indexed)
    }
    torch_save_path = str(output_path)
    torch.save(save_obj, torch_save_path)

    print(f"\n✅ Two-tower model saved to {torch_save_path} (with optimizer/scheduler state for resuming)")


if __name__ == "__main__":
    main()


