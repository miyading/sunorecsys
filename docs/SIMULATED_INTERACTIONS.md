# Simulated User-Item Interactions

## Overview

This system generates realistic simulated user-item interactions to enable collaborative filtering when real interaction data is not available. The simulation creates patterns that mimic real-world behavior:

- **Popular songs get more interactions** (power-law distribution)
- **Users have varying activity levels** (some users are more active)
- **Genre preferences create user clusters** (users tend to like certain genres)
- **Temporal effects** (recent songs slightly more popular)

## Results

With simulated interactions:
- ✅ **94.4% of items** are liked by 2+ users (vs 0% before)
- ✅ **89.3% of items** are liked by 3+ users
- ✅ **100% of users** share items with others
- ✅ **Item-item similarity matrix** has 47.64% non-zero values (vs ~0% before)
- ✅ **Both item-based and user-based CF now work!**

## Two-Tower Training with Simulated Data

The simulated interactions now support generating **positive/negative event pairs** for training two-tower retrieval models with InfoNCE loss.

### Usage for Two-Tower Training

```python
from sunorecsys.data.simulate_interactions import get_user_item_matrix

# Get events with positive/negative pairs
matrix_and_mappings = get_user_item_matrix(
    songs_df=songs_df,
    use_simulation=True,
    return_events=True,              # Enable event generation
    num_negatives_per_user=50,       # Negatives per user
)

user_item_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id, events_df = matrix_and_mappings

# events_df contains: [user_id, song_id, label]
# label=1: positive interaction
# label=0: negative sample (user saw but didn't interact)
```

### Training Script

```bash
python train_two_tower.py \
    --songs data/processed \
    --clap-embeddings data/clap_embeddings.json \
    --output model_checkpoints/two_tower.pt \
    --epochs 5 \
    --batch-size 256 \
    --num-negatives 10
```

### Data Scale Support

The system is optimized for larger datasets:
- **Tested with**: 234 songs, 164 users
- **Designed for**: 3,881 songs, 1,129 users (and beyond)
- **Memory efficient**: Item features stored as float32 (~8MB for 3,881 songs × 512-dim CLAP)
- **Progress tracking**: Automatic progress bars with tqdm for large datasets

## Usage

### Default (Automatic)

The recommenders now use simulated interactions by default:

```python
from sunorecsys.recommenders.item_cf import ItemBasedCFRecommender
from sunorecsys.recommenders.user_based import UserBasedRecommender

# Automatically uses simulated interactions
item_cf = ItemBasedCFRecommender()
item_cf.fit(songs_df)  # Uses simulation by default

user_cf = UserBasedRecommender()
user_cf.fit(songs_df)  # Uses simulation by default
```

### Customize Simulation Parameters

```python
item_cf.fit(
    songs_df,
    use_simulated_interactions=True,
    interaction_rate=0.20,  # Higher = more interactions (0.0 to 1.0)
    random_seed=42,      # For reproducibility
)
```

### Disable Simulation (Use Real Data)

```python
# Option 1: Provide real playlist data
item_cf.fit(
    songs_df,
    playlists=real_playlists,  # Your real playlist data
    use_simulated_interactions=False,
)

# Option 2: Use old user_id-based approach
item_cf.fit(
    songs_df,
    use_simulated_interactions=False,  # Falls back to user_id grouping
)
```

## How to Swap with Real Data

### Method 1: Modify `get_user_item_matrix()` function

Edit `sunorecsys/data/simulate_interactions.py`:

```python
def get_user_item_matrix(
    songs_df: pd.DataFrame,
    playlists: Optional[List[Dict[str, Any]]] = None,
    use_simulation: bool = True,
    ...
):
    if use_simulation or playlists is None:
        # ... existing simulation code ...
    else:
        # ADD YOUR REAL DATA LOADING HERE
        print("📊 Using REAL user-item interactions")
        
        # Load your real interaction data
        # Example: from database, API, file, etc.
        real_interactions = load_real_interactions()  # Your function
        
        # Build matrix from real data
        user_item_matrix, mappings = build_matrix_from_real_data(real_interactions)
        
        return user_item_matrix, mappings
```

### Method 2: Provide Playlist Data

The recommenders already support playlist data:

```python
playlists = [
    {
        'id': 'playlist_1',
        'song_ids': ['song_id_1', 'song_id_2', ...],
        'user_handle': 'user123',  # Optional: creator
    },
    # ... more playlists
]

item_cf.fit(songs_df, playlists=playlists, use_simulated_interactions=False)
```

### Method 3: Direct Matrix Input (Advanced)

Modify the recommender's `fit()` method to accept a pre-built matrix:

```python
# Build your matrix however you want
user_item_matrix = ...  # Your matrix
user_id_to_idx = ...    # Your mappings
# ... etc

# Then set directly (requires modifying recommender code)
item_cf.user_item_matrix = user_item_matrix
item_cf.user_id_to_idx = user_id_to_idx
# ... etc
```

## Simulation Parameters

The `InteractionSimulator` class has several parameters you can tune:

```python
from sunorecsys.data.simulate_interactions import InteractionSimulator

simulator = InteractionSimulator(
    interaction_rate=0.15,        # Overall density (0.0 to 1.0)
    power_law_exponent=2.0,       # Popularity concentration (higher = more concentrated)
    genre_clustering=0.6,         # How much users cluster by genre (0.0 to 1.0)
    temporal_decay=0.1,           # Recent song preference (0.0 to 1.0)
    min_user_interactions=3,      # Minimum interactions per user
    max_user_interactions=50,     # Maximum interactions per user
    random_seed=42,               # For reproducibility
)
```

## Verification

Run the test script to verify everything works:

```bash
python scripts/test_simulated_interactions.py
```

This will:
1. Generate simulated interactions
2. Verify CF viability (should show ✅ now)
3. Train item-based CF and verify it works
4. Show how to swap to real data

## Architecture

```
┌─────────────────────────────────────────┐
│  Recommenders (item_cf, user_based)    │
│  ┌──────────────────────────────────┐ │
│  │ get_user_item_matrix()            │ │
│  │  - use_simulation=True (default)  │ │
│  │  - use_simulation=False (real)    │ │
│  └──────────────────────────────────┘ │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  simulate_interactions.py               │
│  ┌──────────────────────────────────┐ │
│  │ InteractionSimulator              │ │
│  │  - Popularity-based sampling       │ │
│  │  - Genre preferences              │ │
│  │  - User activity levels           │ │
│  └──────────────────────────────────┘ │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Real Data (when available)              │
│  - Playlists                             │
│  - User interactions                     │
│  - Likes/favorites                       │
└─────────────────────────────────────────┘
```

## Benefits

1. **Enables CF immediately**: No need to wait for real interaction data
2. **Realistic patterns**: Mimics real-world behavior (popularity, genre clustering)
3. **Easy to swap**: Single function call to switch to real data
4. **Reproducible**: Random seed ensures consistent results
5. **Configurable**: Adjust parameters to match your domain

## Next Steps

When you have real interaction data:
1. Implement your data loading function
2. Modify `get_user_item_matrix()` to use it
3. Set `use_simulation=False` or provide `playlists` parameter
4. Everything else stays the same!

