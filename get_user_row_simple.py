#!/usr/bin/env python3
"""Simple code snippet to get user row from matrix"""

import joblib
import numpy as np
from pathlib import Path

# Load matrix cache
cache_dir = Path("data/cache")
cache_file = max(cache_dir.glob("user_item_matrix_*.pkl"), key=lambda p: p.stat().st_mtime)
cache_data = joblib.load(cache_file)

# Get mappings
user_item_matrix = cache_data['user_item_matrix']
user_id_to_idx = cache_data['user_id_to_idx']
idx_to_song_id = cache_data['idx_to_song_id']

# Get user row by user_id
user_id = "166970f5-2403-446c-a9df-1f5f32e38d27"  # Replace with your user_id

if user_id in user_id_to_idx:
    user_idx = user_id_to_idx[user_id]
    user_row = user_item_matrix[user_idx].toarray().flatten()
    
    # Get interacted song indices
    interacted_indices = np.where(user_row > 0)[0]
    interacted_song_ids = [idx_to_song_id[idx] for idx in interacted_indices]
    
    print(f"User: {user_id}")
    print(f"Row index: {user_idx}")
    print(f"Interactions: {len(interacted_indices)} songs")
    print(f"Song indices: {list(interacted_indices)}")
    print(f"Song IDs: {interacted_song_ids}")
else:
    print(f"User {user_id} not found")

