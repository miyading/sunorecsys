"""Analyze interaction distributions in original vs synthetic data

This script:
1. Loads original playlist data (if available)
2. Analyzes distribution of item-user interactions in original data
3. Generates synthetic interactions using the simulation
4. Analyzes distribution of interactions in synthetic data
5. Compares the two distributions
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.sparse import csr_matrix

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir.parent))

from sunorecsys.datasets.preprocess import SongDataProcessor
from sunorecsys.datasets.simulate_interactions import (
    get_user_item_matrix,
    build_matrix_from_playlists,
    load_songs_from_aggregated_file,
)


def load_playlists(playlists_file: str):
    """Load playlists from JSON file"""
    with open(playlists_file, 'r') as f:
        data = json.load(f)
    
    playlists = []
    if isinstance(data, list):
        playlists = data
    elif isinstance(data, dict) and 'playlists' in data:
        playlists = data['playlists']
    elif isinstance(data, dict):
        # Try to extract playlists from dict structure
        for key, value in data.items():
            if isinstance(value, list):
                playlists.extend(value)
    
    return playlists


def analyze_matrix_distribution(
    user_item_matrix: csr_matrix,
    user_id_to_idx: dict,
    idx_to_user_id: dict,
    song_id_to_idx: dict,
    idx_to_song_id: dict,
    data_name: str = "Data"
):
    """Analyze and print distribution statistics for a user-item matrix"""
    print(f"\n{'='*80}")
    print(f"{data_name} - Interaction Distribution Analysis")
    print(f"{'='*80}")
    
    num_users = user_item_matrix.shape[0]
    num_items = user_item_matrix.shape[1]
    total_interactions = user_item_matrix.nnz
    
    print(f"\n📊 Basic Statistics:")
    print(f"   Users: {num_users:,}")
    print(f"   Items: {num_items:,}")
    print(f"   Total interactions: {total_interactions:,}")
    print(f"   Sparsity: {(1.0 - total_interactions / (num_users * num_items)) * 100:.2f}%")
    print(f"   Avg interactions per user: {total_interactions / num_users:.2f}")
    print(f"   Avg interactions per item: {total_interactions / num_items:.2f}")
    
    # User interaction distribution
    user_interactions = np.array(user_item_matrix.sum(axis=1)).flatten()
    print(f"\n👤 User Interaction Distribution:")
    print(f"   Min interactions per user: {user_interactions.min()}")
    print(f"   Max interactions per user: {user_interactions.max()}")
    print(f"   Mean interactions per user: {user_interactions.mean():.2f}")
    print(f"   Median interactions per user: {np.median(user_interactions):.2f}")
    print(f"   Std dev: {user_interactions.std():.2f}")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"   Percentiles:")
    for p in percentiles:
        val = np.percentile(user_interactions, p)
        print(f"     {p}th: {val:.1f}")
    
    # Item interaction distribution
    item_interactions = np.array(user_item_matrix.sum(axis=0)).flatten()
    print(f"\n🎵 Item Interaction Distribution:")
    print(f"   Min interactions per item: {item_interactions.min()}")
    print(f"   Max interactions per item: {item_interactions.max()}")
    print(f"   Mean interactions per item: {item_interactions.mean():.2f}")
    print(f"   Median interactions per item: {np.median(item_interactions):.2f}")
    print(f"   Std dev: {item_interactions.std():.2f}")
    
    # Percentiles
    print(f"   Percentiles:")
    for p in percentiles:
        val = np.percentile(item_interactions, p)
        print(f"     {p}th: {val:.1f}")
    
    # Items by interaction count
    item_counts = Counter(item_interactions.astype(int))
    print(f"\n📈 Items by Interaction Count:")
    print(f"   Items with 0 interactions (cold start): {item_counts.get(0, 0)} ({item_counts.get(0, 0)/num_items*100:.1f}%)")
    print(f"   Items with 1 interaction: {item_counts.get(1, 0)} ({item_counts.get(1, 0)/num_items*100:.1f}%)")
    print(f"   Items with 2+ interactions: {sum(v for k, v in item_counts.items() if k >= 2)} ({sum(v for k, v in item_counts.items() if k >= 2)/num_items*100:.1f}%)")
    print(f"   Items with 3+ interactions: {sum(v for k, v in item_counts.items() if k >= 3)} ({sum(v for k, v in item_counts.items() if k >= 3)/num_items*100:.1f}%)")
    print(f"   Items with 5+ interactions: {sum(v for k, v in item_counts.items() if k >= 5)} ({sum(v for k, v in item_counts.items() if k >= 5)/num_items*100:.1f}%)")
    print(f"   Items with 10+ interactions: {sum(v for k, v in item_counts.items() if k >= 10)} ({sum(v for k, v in item_counts.items() if k >= 10)/num_items*100:.1f}%)")
    
    # Users by interaction count
    user_counts = Counter(user_interactions.astype(int))
    print(f"\n👥 Users by Interaction Count:")
    print(f"   Users with 0 interactions: {user_counts.get(0, 0)} ({user_counts.get(0, 0)/num_users*100:.1f}%)")
    print(f"   Users with 1-5 interactions: {sum(v for k, v in user_counts.items() if 1 <= k <= 5)} ({sum(v for k, v in user_counts.items() if 1 <= k <= 5)/num_users*100:.1f}%)")
    print(f"   Users with 6-10 interactions: {sum(v for k, v in user_counts.items() if 6 <= k <= 10)} ({sum(v for k, v in user_counts.items() if 6 <= k <= 10)/num_users*100:.1f}%)")
    print(f"   Users with 11-20 interactions: {sum(v for k, v in user_counts.items() if 11 <= k <= 20)} ({sum(v for k, v in user_counts.items() if 11 <= k <= 20)/num_users*100:.1f}%)")
    print(f"   Users with 21-50 interactions: {sum(v for k, v in user_counts.items() if 21 <= k <= 50)} ({sum(v for k, v in user_counts.items() if 21 <= k <= 50)/num_users*100:.1f}%)")
    print(f"   Users with 50+ interactions: {sum(v for k, v in user_counts.items() if k >= 50)} ({sum(v for k, v in user_counts.items() if k >= 50)/num_users*100:.1f}%)")
    
    # Power-law check: top items
    sorted_items = np.sort(item_interactions)[::-1]
    top_10_pct = int(num_items * 0.1)
    top_10_pct_interactions = sorted_items[:top_10_pct].sum()
    print(f"\n🔝 Popularity Concentration (Power-Law Check):")
    print(f"   Top 10% of items account for {top_10_pct_interactions/total_interactions*100:.1f}% of interactions")
    print(f"   Top 1% of items account for {sorted_items[:int(num_items*0.01)].sum()/total_interactions*100:.1f}% of interactions")
    
    return {
        'num_users': num_users,
        'num_items': num_items,
        'total_interactions': total_interactions,
        'sparsity': 1.0 - total_interactions / (num_users * num_items),
        'avg_interactions_per_user': total_interactions / num_users,
        'avg_interactions_per_item': total_interactions / num_items,
        'user_interactions': user_interactions,
        'item_interactions': item_interactions,
        'item_counts': item_counts,
        'user_counts': user_counts,
    }


def main():
    print("="*80)
    print("Interaction Distribution Analysis: Original vs Synthetic Data")
    print("="*80)
    
    # Step 1: Load songs data
    print("\n📂 Loading songs data...")
    aggregated_file = "sunorecsys/datasets/curl/all_playlist_songs.json"
    songs_df = None
    
    if Path(aggregated_file).exists():
        print(f"   Loading from aggregated file: {aggregated_file}")
        try:
            songs_df = load_songs_from_aggregated_file(aggregated_file, max_songs=None)
            print(f"   ✅ Loaded {len(songs_df)} songs")
        except Exception as e:
            print(f"   ⚠️  Error loading from aggregated file: {e}")
            songs_df = None
    
    if songs_df is None:
        print("   Loading from processed data...")
        processor = SongDataProcessor()
        data_dir = Path("runtime_data/processed")
        if not data_dir.exists():
            print("   ❌ Processed data not found")
            return
        songs_df = processor.load_processed(str(data_dir))
        print(f"   ✅ Loaded {len(songs_df)} songs")
    
    # Step 2: Analyze original data distribution
    # In the original data, each song has a creator (user_id)
    # This represents a form of user-item interaction: user created the song
    print("\n📂 Analyzing original data distribution...")
    print("   Note: Original data shows user-song relationships via song creators")
    print("   (Each song is 'interacted' by its creator)")
    
    # Build a simple matrix: user (creator) x song (created)
    # This represents the original interaction pattern
    user_ids = songs_df['user_id'].unique()
    song_ids = songs_df['song_id'].unique()
    
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    idx_to_user_id = {idx: uid for uid, idx in user_id_to_idx.items()}
    song_id_to_idx = {sid: idx for idx, sid in enumerate(song_ids)}
    idx_to_song_id = {idx: sid for sid, idx in song_id_to_idx.items()}
    
    # Build matrix: user created song = interaction
    rows = []
    cols = []
    data = []
    
    for _, row in songs_df.iterrows():
        user_id = row['user_id']
        song_id = row['song_id']
        if user_id in user_id_to_idx and song_id in song_id_to_idx:
            rows.append(user_id_to_idx[user_id])
            cols.append(song_id_to_idx[song_id])
            data.append(1.0)
    
    original_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(song_ids))
    )
    
    print(f"   ✅ Built matrix from song creators")
    print(f"      Users (creators): {len(user_ids)}")
    print(f"      Songs: {len(song_ids)}")
    print(f"      Interactions (user created song): {len(data)}")
    
    original_stats = analyze_matrix_distribution(
        original_matrix,
        user_id_to_idx,
        idx_to_user_id,
        song_id_to_idx,
        idx_to_song_id,
        "ORIGINAL (Song Creator Data)"
    )
    
    # Step 3: Generate synthetic interactions
    print("\n🎲 Generating synthetic interactions...")
    print("   Simulation parameters:")
    print("     - interaction_rate: 0.15")
    print("     - item_cold_start_rate: 0.05")
    print("     - single_user_item_rate: 0.15")
    print("     - min_user_interactions: 3")
    print("     - max_user_interactions: 50")
    
    synthetic_result = get_user_item_matrix(
        songs_df=songs_df,
        aggregated_file=None,
        playlists=None,
        use_simulation=True,
        interaction_rate=0.15,
        item_cold_start_rate=0.05,
        single_user_item_rate=0.15,
        random_seed=42,
        use_cache=True,
        return_events=False,
    )
    
    synthetic_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id = synthetic_result
    
    synthetic_stats = analyze_matrix_distribution(
        synthetic_matrix,
        user_id_to_idx,
        idx_to_user_id,
        song_id_to_idx,
        idx_to_song_id,
        "SYNTHETIC (Simulated Data)"
    )
    
    # Step 4: Compare if we have both
    if original_stats and synthetic_stats:
        print(f"\n{'='*80}")
        print("COMPARISON: Original vs Synthetic")
        print(f"{'='*80}")
        
        print(f"\n📊 Scale Comparison:")
        print(f"   Users: {original_stats['num_users']:,} → {synthetic_stats['num_users']:,} ({synthetic_stats['num_users']/original_stats['num_users']:.2f}x)")
        print(f"   Items: {original_stats['num_items']:,} → {synthetic_stats['num_items']:,} ({synthetic_stats['num_items']/original_stats['num_items']:.2f}x)")
        print(f"   Interactions: {original_stats['total_interactions']:,} → {synthetic_stats['total_interactions']:,} ({synthetic_stats['total_interactions']/original_stats['total_interactions']:.2f}x)")
        print(f"   Sparsity: {original_stats['sparsity']*100:.2f}% → {synthetic_stats['sparsity']*100:.2f}%")
        
        print(f"\n📈 Distribution Comparison:")
        print(f"   Avg interactions per user: {original_stats['avg_interactions_per_user']:.2f} → {synthetic_stats['avg_interactions_per_user']:.2f}")
        print(f"   Avg interactions per item: {original_stats['avg_interactions_per_item']:.2f} → {synthetic_stats['avg_interactions_per_item']:.2f}")
        
        # Cold start items
        orig_cold = original_stats['item_counts'].get(0, 0)
        synth_cold = synthetic_stats['item_counts'].get(0, 0)
        print(f"\n   Cold Start Items (0 interactions):")
        print(f"     Original: {orig_cold} ({orig_cold/original_stats['num_items']*100:.1f}%)")
        print(f"     Synthetic: {synth_cold} ({synth_cold/synthetic_stats['num_items']*100:.1f}%)")
        
        # Single-user items
        orig_single = original_stats['item_counts'].get(1, 0)
        synth_single = synthetic_stats['item_counts'].get(1, 0)
        print(f"\n   Single-User Items (1 interaction):")
        print(f"     Original: {orig_single} ({orig_single/original_stats['num_items']*100:.1f}%)")
        print(f"     Synthetic: {synth_single} ({synth_single/synthetic_stats['num_items']*100:.1f}%)")
        
        # Multi-user items
        orig_multi = sum(v for k, v in original_stats['item_counts'].items() if k >= 2)
        synth_multi = sum(v for k, v in synthetic_stats['item_counts'].items() if k >= 2)
        print(f"\n   Multi-User Items (2+ interactions):")
        print(f"     Original: {orig_multi} ({orig_multi/original_stats['num_items']*100:.1f}%)")
        print(f"     Synthetic: {synth_multi} ({synth_multi/synthetic_stats['num_items']*100:.1f}%)")
    
    # Step 5: Explain what the simulation does
    print(f"\n{'='*80}")
    print("WHAT THE SIMULATION CODE DOES")
    print(f"{'='*80}")
    print("""
The simulation code (InteractionSimulator) generates realistic user-item interactions:

1. **Power-Law Popularity Distribution**
   - Popular songs get more interactions
   - Uses popularity_score from songs_df if available
   - Applies power-law transformation (exponent=2.0 by default)

2. **User Activity Levels**
   - Users have varying activity levels (Beta distribution)
   - Each user gets 3-50 interactions (configurable)
   - Ensures no user cold-start (all users have minimum interactions)

3. **Genre Clustering**
   - Users prefer genres from their created songs (genre_clustering=0.6)
   - But also explore other genres (40% exploration)
   - Creates user clusters based on genre preferences

4. **Temporal Effects**
   - Recent songs get slight boost (temporal_decay=0.1)
   - Uses days_since_creation if available

5. **Item Distribution Control**
   - item_cold_start_rate (5%): Items with zero interactions
   - single_user_item_rate (15%): Items with exactly one user
   - Remaining items: Multi-user items (for CF to work)

6. **Sampling Process**
   - First pass: Assign interactions to users (excluding cold-start items)
   - Second pass: Assign single-user items to random users
   - Cold-start items remain with 0 interactions (use content-based fallback)

7. **Optional Event Generation**
   - Can generate positive/negative event pairs for two-tower training
   - Positive: All user-item interactions
   - Negative: Sampled from non-interacted items (50 per user by default)
    """)
    
    print(f"\n{'='*80}")
    print("✅ Analysis Complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

