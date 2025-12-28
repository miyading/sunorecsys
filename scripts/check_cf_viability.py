"""Check if the data is viable for collaborative filtering"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.sparse import csr_matrix

from sunorecsys.datasets.preprocess import SongDataProcessor
from sunorecsys.recommenders.item_cf import ItemBasedCFRecommender


def analyze_cf_viability(songs_df: pd.DataFrame, user_item_matrix: csr_matrix, 
                         user_id_to_idx: dict, song_id_to_idx: dict):
    """Analyze if the data supports collaborative filtering"""
    
    print("\n" + "="*80)
    print("Collaborative Filtering Viability Analysis")
    print("="*80)
    
    num_users, num_items = user_item_matrix.shape
    
    # Check item co-occurrence (critical for item-based CF)
    print("\n1. Item Co-occurrence Analysis (for Item-Based CF):")
    print("-" * 80)
    
    item_interaction_counts = np.array(user_item_matrix.sum(axis=0)).flatten()
    items_with_multiple_users = np.sum(item_interaction_counts > 1)
    items_with_one_user = np.sum(item_interaction_counts == 1)
    
    print(f"   Items liked by 1 user:     {items_with_one_user} ({items_with_one_user/num_items*100:.1f}%)")
    print(f"   Items liked by 2+ users:   {items_with_multiple_users} ({items_with_multiple_users/num_items*100:.1f}%)")
    print(f"   Items liked by 3+ users:   {np.sum(item_interaction_counts >= 3)} ({np.sum(item_interaction_counts >= 3)/num_items*100:.1f}%)")
    print(f"   Items liked by 5+ users:   {np.sum(item_interaction_counts >= 5)} ({np.sum(item_interaction_counts >= 5)/num_items*100:.1f}%)")
    
    if items_with_multiple_users == 0:
        print("\n   ❌ PROBLEM: No items are liked by multiple users!")
        print("   → Item-based CF will NOT work (no item co-occurrence)")
        print("   → Item-item similarity matrix will be all zeros")
    elif items_with_multiple_users < num_items * 0.1:
        print("\n   ⚠️  WARNING: Very few items are liked by multiple users")
        print("   → Item-based CF will have limited effectiveness")
    else:
        print("\n   ✅ Good: Many items are liked by multiple users")
        print("   → Item-based CF should work")
    
    # Check user overlap (critical for user-based CF)
    print("\n2. User Overlap Analysis (for User-Based CF):")
    print("-" * 80)
    
    user_interaction_counts = np.array(user_item_matrix.sum(axis=1)).flatten()
    
    # Compute user-user overlap
    user_user_overlaps = user_item_matrix @ user_item_matrix.T  # Dot product = shared items
    # Remove diagonal (self-overlap)
    np.fill_diagonal(user_user_overlaps.toarray(), 0)
    
    # Count users with at least one overlapping item with another user
    users_with_overlap = np.sum((user_user_overlaps > 0).sum(axis=1) > 0)
    users_without_overlap = num_users - users_with_overlap
    
    print(f"   Users with overlapping items: {users_with_overlap} ({users_with_overlap/num_users*100:.1f}%)")
    print(f"   Users with NO overlap:        {users_without_overlap} ({users_without_overlap/num_users*100:.1f}%)")
    
    # Average overlap per user pair
    total_overlaps = user_user_overlaps.sum()
    num_user_pairs = num_users * (num_users - 1) / 2
    avg_overlap = total_overlaps / num_user_pairs if num_user_pairs > 0 else 0
    
    print(f"   Average shared items per user pair: {avg_overlap:.3f}")
    
    if users_with_overlap == 0:
        print("\n   ❌ PROBLEM: No users share any items!")
        print("   → User-based CF will NOT work (no user similarity)")
        print("   → Cannot find similar users to make recommendations")
    elif users_with_overlap < num_users * 0.1:
        print("\n   ⚠️  WARNING: Very few users have overlapping items")
        print("   → User-based CF will have limited effectiveness")
    else:
        print("\n   ✅ Good: Many users share items with others")
        print("   → User-based CF should work")
    
    # Check matrix structure
    print("\n3. Matrix Structure Analysis:")
    print("-" * 80)
    
    # Check if it's a permutation matrix (each row has exactly one 1, each column has exactly one 1)
    rows_with_one = np.sum(user_interaction_counts == 1)
    cols_with_one = np.sum(item_interaction_counts == 1)
    
    is_permutation = (rows_with_one == num_users) and (cols_with_one == num_items)
    
    if is_permutation:
        print("   ❌ PROBLEM: Matrix is a PERMUTATION MATRIX!")
        print("   → Each user has exactly 1 item, each item belongs to exactly 1 user")
        print("   → This means NO co-occurrence, NO overlap")
        print("   → Collaborative filtering CANNOT work with this structure")
        print("\n   Why this happens:")
        print("   - Each song has a unique user_id (creator)")
        print("   - Songs don't appear in playlists or multiple users' collections")
        print("   - This is NOT collaborative data - it's just ownership data")
    else:
        print(f"   ✅ Matrix is NOT a permutation matrix")
        print(f"   → Rows with 1 item: {rows_with_one}/{num_users} ({rows_with_one/num_users*100:.1f}%)")
        print(f"   → Cols with 1 user: {cols_with_one}/{num_items} ({cols_with_one/num_items*100:.1f}%)")
    
    # Recommendations
    print("\n" + "="*80)
    print("Recommendations:")
    print("="*80)
    
    if is_permutation or items_with_multiple_users == 0:
        print("\n❌ Your current data structure does NOT support collaborative filtering.")
        print("\nTo make collaborative filtering work, you need:")
        print("  1. Songs that appear in MULTIPLE playlists/users")
        print("  2. Users who share songs with other users")
        print("  3. Co-occurrence patterns (songs that appear together)")
        print("\nSolutions:")
        print("  A. Use PLAYLIST data where:")
        print("     - Songs can appear in multiple playlists")
        print("     - Multiple users can have the same songs")
        print("     - Playlists represent user preferences")
        print("\n  B. Use USER INTERACTION data where:")
        print("     - Users can like/favorite multiple songs")
        print("     - Songs can be liked by multiple users")
        print("     - There's overlap between users' preferences")
        print("\n  C. For now, use CONTENT-BASED methods instead:")
        print("     - Item content-based (embeddings + genre)")
        print("     - Prompt-based similarity")
        print("     - These don't require user-item interactions")
    else:
        print("\n✅ Your data structure supports collaborative filtering!")
        print("   You can use both item-based and user-based CF.")
    
    return {
        'is_viable': not is_permutation and items_with_multiple_users > 0,
        'is_permutation_matrix': is_permutation,
        'items_with_multiple_users': int(items_with_multiple_users),
        'users_with_overlap': int(users_with_overlap),
    }


def main():
    print("="*80)
    print("Checking Collaborative Filtering Viability")
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
    
    # Build matrix using current approach
    print("\n🔧 Building user-item matrix...")
    item_cf = ItemBasedCFRecommender()
    item_cf.fit(songs_df)
    
    # Analyze viability
    results = analyze_cf_viability(
        songs_df,
        item_cf.user_item_matrix,
        item_cf.user_id_to_idx,
        item_cf.song_id_to_idx
    )
    
    # Show what the data actually represents
    print("\n" + "="*80)
    print("What Your Data Represents:")
    print("="*80)
    
    if 'user_id' in songs_df.columns:
        unique_users = songs_df['user_id'].nunique()
        songs_per_user = songs_df.groupby('user_id').size()
        
        print(f"\nYour songs have 'user_id' field:")
        print(f"  - {unique_users} unique users")
        print(f"  - {songs_per_user.mean():.2f} songs per user (avg)")
        print(f"  - {songs_per_user.min()} - {songs_per_user.max()} songs per user (range)")
        
        print("\nThis 'user_id' likely represents:")
        print("  - The CREATOR of the song (who generated it)")
        print("  - NOT the user who LIKES/SAVED the song")
        print("  - This is ownership data, not preference data")
        
        print("\nFor collaborative filtering, you need:")
        print("  - Users who LIKE/SAVE songs (not create them)")
        print("  - Songs that appear in multiple users' collections")
        print("  - Playlist data where songs can be added by users")
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

