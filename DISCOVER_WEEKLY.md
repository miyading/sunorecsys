# Discover Weekly Implementation Guide

## Overview

This system implements a Discover Weekly-style recommendation system similar to Spotify, with weekly updates and user interaction history management.

## Key Features

### 1. User History Management (`user_history.py`)

- **Last-n interactions**: Tracks last N interactions per user (default: 50)
- **Weekly mixing**: 
  - If user has no interactions this week → use all historical interactions
  - If user has interactions this week → 50% historical + 50% this week's new interactions
- **Weighted interactions**: Supports play counts and other interaction weights
- **Weekly updates**: Models update every Monday (configurable)

### 2. Item-Based CF (Discover Weekly Style)

**Current Implementation:**
- ✅ Precomputed item-item similarity matrix
- ✅ User-to-item indexing
- ✅ Weighted similarity (if play counts available)
- ✅ Top-k per seed song, then deduplicate
- ✅ Weekly model updates

**How it works:**
1. Get user's last-n interactions (with weekly mixing)
2. For each seed song, find top-k most similar items (default: 20)
3. Aggregate scores across all seed songs
4. Deduplicate and return top N items

### 3. Other Channels

**Item Content-Based & Prompt-Based:**
- ✅ Support `use_last_n` parameter
- ✅ Can use user history if provided
- ✅ Default uses last-n interactions for Discover Weekly style

---

## Integration Status

### ✅ Completed Features

#### 1. **HybridRecommender Automatic Integration**

**How it works:**
- `HybridRecommender` automatically creates and manages `UserHistoryManager`
- When `user_id` is provided, automatically fetches last-n interactions from history
- Supports weekly mixing (50/50 historical + this week's new interactions)

**Usage:**
```python
# Create with automatic integration
recommender = HybridRecommender(
    history_manager=history_manager,  # Optional, auto-created if not provided
    use_last_n=True,  # Default: enabled
)

# Recommend - just provide user_id
recommendations = recommender.recommend(
    user_id="user123",  # Automatically uses last-n interactions
    n=10,
)
```

#### 2. **All Channels Default to Last-N**

**Item-Based CF:**
- ✅ Uses last-n interactions as seed songs
- ✅ Top-k-per-seed=5 (finds top-5 similar for each seed song)
- ✅ Deduplicates and returns top items
- ✅ Supports weighted similarity (if play counts available)

**Item Content-Based:**
- ✅ Default uses last-n (`use_last_n=True`)
- ✅ Uses embedding similarity (content-based)

**Prompt-Based:**
- ✅ Default uses last-n (`use_last_n=True`)
- ✅ Uses prompt embedding similarity (content-based)

**User-Based CF:**
- ✅ Uses user_id directly for recommendations (doesn't need seed songs)

#### 3. **Weekly Automatic Updates**

**Created `weekly_update.py` script:**
- Checks if update is needed (every Monday)
- Retrains all models
- Updates item-item similarity matrix
- Saves updated models

**Usage Methods:**

**Method 1: Cron Job (Recommended)**
```bash
# Add to crontab
0 0 * * 1 cd /path/to/sunorecsys && python weekly_update.py
# Runs every Monday at midnight
```

**Method 2: Python Scheduler**
```python
from sunorecsys.utils.weekly_update import WeeklyUpdateScheduler

scheduler = WeeklyUpdateScheduler(
    recommender=recommender,
    history_manager=history_manager,
    songs_df=songs_df,
)

# Run once if update is needed
scheduler.run_once_if_needed()

# Or run continuously (check every 60 seconds)
scheduler.run_forever(interval=60)
```

---

## Usage

### Setting Up User History

```python
from sunorecsys.data.user_history import UserHistoryManager

# Initialize manager
history_manager = UserHistoryManager(
    history_file="data/user_history.json",
    last_n=50,  # Keep last 50 interactions per user
    weekly_update_day=0,  # Monday
)

# Add interactions
history_manager.add_interaction(
    user_id="user123",
    song_id="song456",
    timestamp=datetime.now(),
    weight=3.0,  # Play count
    interaction_type="play"
)

# Get user's interactions for recommendations
interactions = history_manager.get_user_interactions(
    user_id="user123",
    use_weekly_mixing=True  # 50/50 mix if has this week's interactions
)
```

### Using with Recommenders

```python
from sunorecsys.recommenders.hybrid import HybridRecommender
from sunorecsys.data.user_history import UserHistoryManager

# Initialize (history manager is optional - auto-created if not provided)
history_manager = UserHistoryManager()
recommender = HybridRecommender(
    history_manager=history_manager,
    use_last_n=True,  # Default: enabled
)

# Fit with weighted interactions (if available)
recommender.fit(
    songs_df,
    use_simulated_interactions=True,
    # If you have play counts, they'll be used for weighted similarity
)

# Get recommendations using user history (automatic)
recommendations = recommender.recommend(
    user_id="user123",  # Automatically uses last-n interactions
    n=30,  # Discover Weekly typically has 30 songs
)
```

### Weekly Model Updates

```python
# Check if models need updating (every Monday)
if history_manager.should_update_models():
    print("Updating models for new week...")
    
    # Re-fit recommenders with new data
    recommender.fit(songs_df, ...)
    # ... update other channels
    
    print("✅ Models updated for this week")
```

---

## Implementation Details

### Item-CF Implementation ✅

```python
# 1. Get user's last-n interactions (with weekly mixing)
# HybridRecommender automatically handles this

# 2. For each seed song, find top-5 most similar
item_cf.recommend(
    song_ids=interactions,  # Last-n from history
    top_k_per_seed=5,  # Find top-5 for each seed
    n=10,  # Final return 10 songs
)

# 3. Automatically deduplicates and returns top items
# 4. If play counts available, uses weighted similarity
```

### Other Channels ✅

- **Item Content-Based**: Uses embedding similarity, defaults to last-n
- **Prompt-Based**: Uses prompt similarity, defaults to last-n
- **User-Based CF**: Uses collaborative similarity, based on user_id

---

## Configuration

**Current Settings:**
- `top_k_per_seed=5` (Item-CF)
- `n=10` (final recommendation count)
- `use_last_n=True` (default enabled)
- `last_n=50` (default number of interactions to track)

**Modify Configuration:**
   ```python
recommender = HybridRecommender(
    use_last_n=True,  # Enable last-n
)

# Can override when recommending
recommendations = recommender.recommend(
    user_id=user_id,
    n=10,  # Final count
    # Item-CF internally uses top_k_per_seed=5
)
```

---

## Workflow

### Daily Recommendation Generation
```python
# 1. User has new interactions
history_manager.add_interaction(user_id, song_id, weight=play_count)

# 2. Generate recommendations (automatically uses last-n)
recommendations = recommender.recommend(user_id=user_id, n=10)
```

### Weekly Model Updates
   ```python
# Runs automatically every Monday
python weekly_update.py

# Or check manually if update is needed
   if history_manager.should_update_models():
    recommender.fit(songs_df)  # Retrain
   ```

---

## Complete Example: Full Discover Weekly Flow

```python
from sunorecsys.data.user_history import UserHistoryManager
from sunorecsys.recommenders.hybrid import HybridRecommender

# Initialize
history_manager = UserHistoryManager()
recommender = HybridRecommender(
    history_manager=history_manager,
    use_last_n=True,
)
recommender.fit(songs_df, ...)

# Weekly recommendation generation
def generate_discover_weekly(user_id: str, n: int = 30):
    # Get user's last-n interactions (with weekly mixing)
    # This happens automatically when user_id is provided
    recommendations = recommender.recommend(
        user_id=user_id,  # Automatically uses last-n from history
        n=n,
    )
    
    return recommendations

# Check if models need updating
if history_manager.should_update_models():
    # Re-fit models with new data
    recommender.fit(songs_df, ...)
```

---

## File Structure

```
sunorecsys/
├── data/
│   └── user_history.py          # User history management
├── recommenders/
│   ├── hybrid.py                # Integrated with UserHistoryManager
│   ├── item_cf.py               # Top-k-per-seed=5, weighted similarity
│   ├── item_content_based.py    # Default use_last_n=True
│   └── prompt_based.py          # Default use_last_n=True
├── utils/
│   └── weekly_update.py         # Weekly update scheduler
├── weekly_update.py             # Executable update script
└── discover_weekly_example.py   # Complete example
```

---

## Summary

All core features are implemented! The system now:
- ✅ Automatically uses last-n interactions
- ✅ Supports weekly mixing (50/50 historical + this week)
- ✅ Top-k-per-seed=5 for Item-CF
- ✅ Weekly automatic updates
- ✅ All channels default to using last-n

Ready for production use!
