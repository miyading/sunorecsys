"""Script to visualize and analyze the user-item interaction matrix"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from sunorecsys.datasets.preprocess import SongDataProcessor
from sunorecsys.recommenders.item_cf import ItemBasedCFRecommender


def load_playlists(playlists_file: str):
    """Load playlist data"""
    with open(playlists_file, 'r') as f:
        data = json.load(f)
    
    playlists = []
    if 'result' in data and 'playlist' in data['result']:
        playlist_data = data['result']['playlist'].get('result', [])
        for playlist in playlist_data:
            playlists.append({
                'id': playlist.get('id'),
                'name': playlist.get('name'),
                'user_handle': playlist.get('user_handle'),  # Actual creator
                'user_display_name': playlist.get('user_display_name'),
                'song_ids': playlist.get('playlist_clips', []),  # Would need actual song IDs
            })
    
    return playlists


def analyze_matrix(matrix: csr_matrix, user_id_to_idx: dict, song_id_to_idx: dict, 
                   idx_to_user_id: dict, idx_to_song_id: dict, title: str):
    """Analyze and visualize the user-item matrix"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # Basic statistics
    num_users, num_items = matrix.shape
    num_interactions = matrix.nnz  # Number of non-zero entries
    sparsity = 1.0 - (num_interactions / (num_users * num_items))
    
    print(f"\nMatrix Shape: ({num_users} users, {num_items} items)")
    print(f"Total Interactions: {num_interactions:,}")
    print(f"Sparsity: {sparsity:.4%} ({(1-sparsity)*100:.2f}% dense)")
    print(f"Average interactions per user: {num_interactions/num_users:.2f}")
    print(f"Average interactions per item: {num_interactions/num_items:.2f}")
    
    # User statistics
    user_interaction_counts = np.array(matrix.sum(axis=1)).flatten()
    print(f"\nUser Interaction Statistics:")
    print(f"  Min interactions per user: {user_interaction_counts.min()}")
    print(f"  Max interactions per user: {user_interaction_counts.max()}")
    print(f"  Mean interactions per user: {user_interaction_counts.mean():.2f}")
    print(f"  Median interactions per user: {np.median(user_interaction_counts):.2f}")
    
    # Item statistics
    item_interaction_counts = np.array(matrix.sum(axis=0)).flatten()
    print(f"\nItem Interaction Statistics:")
    print(f"  Min interactions per item: {item_interaction_counts.min()}")
    print(f"  Max interactions per item: {item_interaction_counts.max()}")
    print(f"  Mean interactions per item: {item_interaction_counts.mean():.2f}")
    print(f"  Median interactions per item: {np.median(item_interaction_counts):.2f}")
    
    # Show sample of the matrix
    print(f"\n{'='*80}")
    print("Sample Matrix Entries (first 10 users, first 20 items):")
    print(f"{'='*80}")
    
    sample_matrix = matrix[:10, :20].toarray()
    df_sample = pd.DataFrame(
        sample_matrix,
        index=[idx_to_user_id.get(i, f"user_{i}")[:20] for i in range(min(10, num_users))],
        columns=[idx_to_song_id.get(i, f"song_{i}")[:8] for i in range(min(20, num_items))]
    )
    print(df_sample.to_string())
    
    # Top users by interactions
    print(f"\n{'='*80}")
    print("Top 10 Users by Number of Interactions:")
    print(f"{'='*80}")
    top_user_indices = np.argsort(user_interaction_counts)[::-1][:10]
    for rank, user_idx in enumerate(top_user_indices, 1):
        user_id = idx_to_user_id.get(user_idx, f"user_{user_idx}")
        count = int(user_interaction_counts[user_idx])
        print(f"{rank:2d}. {user_id[:50]:50s} - {count:4d} interactions")
    
    # Top items by interactions
    print(f"\n{'='*80}")
    print("Top 10 Items by Number of Interactions:")
    print(f"{'='*80}")
    top_item_indices = np.argsort(item_interaction_counts)[::-1][:10]
    for rank, item_idx in enumerate(top_item_indices, 1):
        item_id = idx_to_song_id.get(item_idx, f"song_{item_idx}")
        count = int(item_interaction_counts[item_idx])
        print(f"{rank:2d}. {item_id[:50]:50s} - {count:4d} interactions")
    
    return {
        'shape': (num_users, num_items),
        'sparsity': sparsity,
        'num_interactions': num_interactions,
        'user_stats': {
            'min': int(user_interaction_counts.min()),
            'max': int(user_interaction_counts.max()),
            'mean': float(user_interaction_counts.mean()),
            'median': float(np.median(user_interaction_counts)),
        },
        'item_stats': {
            'min': int(item_interaction_counts.min()),
            'max': int(item_interaction_counts.max()),
            'mean': float(item_interaction_counts.mean()),
            'median': float(np.median(item_interaction_counts)),
        }
    }


def visualize_matrix(matrix: csr_matrix, title: str, save_path: str = None):
    """Visualize the user-item matrix"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib not available. Skipping visualization.")
        return
    
    # Sample a subset for visualization (if too large)
    max_size = 100
    if matrix.shape[0] > max_size or matrix.shape[1] > max_size:
        print(f"\nMatrix too large for visualization ({matrix.shape}). Sampling first {max_size}x{max_size}...")
        matrix_sample = matrix[:max_size, :max_size].toarray()
    else:
        matrix_sample = matrix.toarray()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(matrix_sample, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Interaction (1 = present, 0 = absent)')
    plt.title(f'{title}\nUser-Item Interaction Matrix')
    plt.xlabel('Items (Songs)')
    plt.ylabel('Users')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    else:
        plt.show()


def build_matrix_with_playlist_creators(songs_df: pd.DataFrame, playlists: list):
    """
    Build user-item matrix where actual playlist creators are the users.
    This handles the case where playlists have creators (user_handle/user_id).
    
    Approach:
    1. If a playlist has a creator (user_handle), use that as the user
    2. If multiple playlists are created by the same user, aggregate their songs
    3. This gives a true user-item matrix based on actual user behavior
    """
    print("\n" + "="*80)
    print("Building User-Item Matrix with Playlist Creators")
    print("="*80)
    
    rows = []
    cols = []
    data = []
    
    song_id_to_idx = {}
    user_id_to_idx = {}
    idx_to_song_id = {}
    idx_to_user_id = {}
    
    # Create song mappings
    for song_id in songs_df['song_id'].unique():
        if song_id not in song_id_to_idx:
            idx = len(song_id_to_idx)
            song_id_to_idx[song_id] = idx
            idx_to_song_id[idx] = song_id
    
    all_song_ids = set(songs_df['song_id'].unique())
    
    # Build user-item interactions from playlists
    user_playlists = {}  # Track which playlists belong to which users
    
    for playlist in playlists:
        playlist_id = playlist.get('id')
        song_ids = playlist.get('song_ids', [])
        
        # Filter to known songs
        song_ids = [sid for sid in song_ids if sid in all_song_ids]
        if len(song_ids) == 0:
            continue
        
        # Get the creator (user) of this playlist
        # Priority: user_handle > user_display_name > playlist_id (fallback)
        creator_id = (
            playlist.get('user_handle') or 
            playlist.get('user_display_name') or 
            f"playlist_{playlist_id}"  # Fallback: treat playlist as user
        )
        
        # Track playlists per user
        if creator_id not in user_playlists:
            user_playlists[creator_id] = []
        user_playlists[creator_id].append({
            'playlist_id': playlist_id,
            'song_ids': song_ids
        })
        
        # Create user mapping if needed
        if creator_id not in user_id_to_idx:
            user_idx = len(user_id_to_idx)
            user_id_to_idx[creator_id] = user_idx
            idx_to_user_id[user_idx] = creator_id
        
        user_idx = user_id_to_idx[creator_id]
        
        # Add interactions for songs in this playlist
        for song_id in song_ids:
            if song_id in song_id_to_idx:
                song_idx = song_id_to_idx[song_id]
                rows.append(user_idx)
                cols.append(song_idx)
                data.append(1.0)  # Binary interaction
    
    # Build sparse matrix
    user_item_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_id_to_idx), len(song_id_to_idx))
    )
    
    print(f"✅ Built matrix with {len(user_id_to_idx)} users (playlist creators)")
    print(f"   Total playlists: {len(playlists)}")
    print(f"   Users with multiple playlists: {sum(1 for v in user_playlists.values() if len(v) > 1)}")
    
    # Show users with multiple playlists
    multi_playlist_users = {k: len(v) for k, v in user_playlists.items() if len(v) > 1}
    if multi_playlist_users:
        print(f"\nUsers with multiple playlists (top 10):")
        sorted_users = sorted(multi_playlist_users.items(), key=lambda x: x[1], reverse=True)[:10]
        for user_id, count in sorted_users:
            print(f"  {user_id[:50]:50s} - {count} playlists")
    
    return user_item_matrix, user_id_to_idx, song_id_to_idx, idx_to_user_id, idx_to_song_id


def main():
    print("="*80)
    print("User-Item Interaction Matrix Analysis")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    processor = SongDataProcessor()
    
    data_dir = Path("runtime_data/processed")
    if not data_dir.exists():
        print("❌ Processed data not found. Please run preprocessing first.")
        return
    
    try:
        songs_df = processor.load_processed(str(data_dir))
        print(f"✅ Loaded {len(songs_df)} songs")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Method 1: Current approach - Each playlist = a "user"
    print("\n" + "="*80)
    print("Method 1: Current Approach (Each Playlist = A User)")
    print("="*80)
    print("\nThis approach treats each playlist as a separate 'user'.")
    print("This is a simplification that works when:")
    print("  - You don't have playlist creator information")
    print("  - You want to use playlist co-occurrence patterns")
    print("  - Each playlist represents a distinct taste profile")
    print("\nLimitations:")
    print("  - Doesn't track actual users")
    print("  - Can't aggregate multiple playlists by the same user")
    print("  - May create duplicate 'users' if same person has multiple playlists")
    
    item_cf = ItemBasedCFRecommender()
    item_cf.fit(songs_df)  # Uses _build_matrix_from_users by default
    
    stats1 = analyze_matrix(
        item_cf.user_item_matrix,
        item_cf.user_id_to_idx,
        item_cf.song_id_to_idx,
        item_cf.idx_to_user_id,
        item_cf.idx_to_song_id,
        "Current Approach: Playlist = User"
    )
    
    # Method 2: Using actual playlist creators (if available)
    print("\n" + "="*80)
    print("Method 2: Using Actual Playlist Creators")
    print("="*80)
    print("\nThis approach uses the actual creator (user_handle) of each playlist.")
    print("Benefits:")
    print("  - Tracks real users, not just playlists")
    print("  - Aggregates all playlists by the same user")
    print("  - More accurate user profiles")
    print("\nWhen to use:")
    print("  - You have playlist creator information (user_handle, user_id)")
    print("  - You want to recommend to actual users")
    print("  - Users can have multiple playlists")
    
    # Try to load playlists if available
    playlists_file = Path("results.json")
    playlists = []
    if playlists_file.exists():
        try:
            playlists = load_playlists(str(playlists_file))
            print(f"\n✅ Loaded {len(playlists)} playlists from {playlists_file}")
        except Exception as e:
            print(f"⚠️  Could not load playlists: {e}")
            print("   Using user_id from songs_df instead...")
    
    if playlists:
        # Build matrix with actual creators
        matrix2, user_id_to_idx2, song_id_to_idx2, idx_to_user_id2, idx_to_song_id2 = \
            build_matrix_with_playlist_creators(songs_df, playlists)
        
        stats2 = analyze_matrix(
            matrix2,
            user_id_to_idx2,
            song_id_to_idx2,
            idx_to_user_id2,
            idx_to_song_id2,
            "Creator-Based Approach: Actual Users"
        )
        
        # Comparison
        print("\n" + "="*80)
        print("Comparison")
        print("="*80)
        print(f"\nMethod 1 (Playlist=User):")
        print(f"  Users: {stats1['shape'][0]}")
        print(f"  Items: {stats1['shape'][1]}")
        print(f"  Sparsity: {stats1['sparsity']:.4%}")
        
        print(f"\nMethod 2 (Creator-Based):")
        print(f"  Users: {stats2['shape'][0]}")
        print(f"  Items: {stats2['shape'][1]}")
        print(f"  Sparsity: {stats2['sparsity']:.4%}")
        
        print(f"\nDifference:")
        print(f"  User reduction: {stats1['shape'][0] - stats2['shape'][0]} "
              f"({(1 - stats2['shape'][0]/stats1['shape'][0])*100:.1f}% fewer)")
    else:
        print("\n⚠️  No playlist data available. Showing only Method 1.")
        print("\nTo use Method 2, you need playlist data with creator information.")
        print("The playlist data should include:")
        print("  - 'user_handle' or 'user_id': The creator of the playlist")
        print("  - 'song_ids' or 'playlist_clips': Songs in the playlist")
    
    # Visualization (optional)
    try:
        visualize_matrix(
            item_cf.user_item_matrix,
            "Current Approach",
            save_path="user_item_matrix_visualization.png"
        )
    except Exception as e:
        print(f"\n⚠️  Could not create visualization: {e}")
        print("   (This is optional - install matplotlib to enable)")
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

