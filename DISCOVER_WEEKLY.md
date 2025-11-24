# Discover Weekly Implementation & System Design

## Overview

This system implements a Discover Weekly-style recommendation system similar to Spotify, with weekly updates and user interaction history management. The system employs a multi-stage hybrid architecture for music recommendation that uniquely leverages the characteristics of AI music generation platforms.

## System Design Summary

The system employs a multi-stage hybrid architecture for music recommendation that uniquely leverages the characteristics of AI music generation platforms. In the recall stage, Channel 3 (Two-Tower) uses CLAP audio embeddings for content retrieval, while Channel 6 (Prompt-Based) in the fine ranking stage uses CLAP text embeddings of user generation prompts to capture creative intent—a signal unique to AI music platforms where prompts directly generate music. By using CLAP's aligned text-audio embedding space, prompt embeddings can be directly compared with audio embeddings without cross-modal alignment, enabling unified similarity search that matches user creative intent with audio characteristics. This design allows the prompt-based and CLAP-based channels to potentially be combined into a single multi-modal content channel, as both operate in the same aligned embedding space. The system also implements a Deep Interest Network (DIN) in the fine ranking stage for CTR prediction, using attention mechanisms to aggregate user's historical track embeddings and predict click-through rates for candidate tracks. This attention-based aggregation shares the same architectural principles that could be applied to hierarchical embedding aggregation (tracks → artists), addressing the research question of representing higher-level embeddings from lower-level ones through learned aggregation strategies beyond simple averaging.

### Key Design Features

1. **Multi-Modal Alignment**: CLAP provides aligned text-audio embeddings, enabling direct similarity between prompts and audio
2. **DIN for CTR Prediction**: Attention-based aggregation of user history for personalized ranking
3. **Hierarchical Aggregation Potential**: Same attention mechanism can aggregate tracks → artists
4. **AI Music Platform Specificity**: Leverages prompt-based creative intent as a unique signal
5. **Unified Embedding Space**: Text and audio embeddings in same space enable unified content-based retrieval

---

## Four-Stage Architecture

### ✅ Stage 1 (Recall): Candidate Retrieval

**Status**: ✅ **COMPLETE**

- ✅ **Channel 1: Item-based CF** - User-item interaction matrix based similarity (top-k per seed)
- ✅ **Channel 2: User-based CF** - Matrix factorization (ALS) based on user-item interactions  
- ✅ **Channel 3: Two-tower model** - CLAP-based content retrieval (audio embeddings)

**Implementation Details**:
- Item-based CF uses precomputed item-item similarity matrix
- User-based CF uses ALS matrix factorization
- Two-tower model uses CLAP audio embeddings for content retrieval
- All channels support last-n interactions for Discover Weekly style

### ✅ Stage 2 (Coarse Ranking): Quality Filter

**Status**: ✅ **COMPLETE**

- ✅ **Channel 4: Quality filter** - Engagement-based quality scoring

**Implementation Details**:
- Filters songs below quality threshold (default: 0.3)
- Uses engagement metrics (plays, upvotes, comments)
- Currently uses heuristic-based scoring
- ⚠️ **TODO**: Integrate Meta Audiobox Aesthetics for better quality scoring

### ✅ Stage 3 (Fine Ranking): CTR Prediction

**Status**: ✅ **COMPLETE**

- ✅ **Channel 5: DIN with attention** - CTR prediction using user history
- ✅ **Channel 6: Prompt-based similarity** - CLAP text embeddings (aligned with audio)

**Implementation Details**:
- DIN uses attention to aggregate user's historical track embeddings
- Predicts click-through rate for candidate tracks
- Prompt-based uses CLAP text embeddings (aligned with audio space)
- Both operate in the same CLAP embedding space

### ⚠️ Stage 4 (Re-ranking): Final Ranking

**Status**: ⚠️ **PARTIAL** (Optional, computationally intensive)

- ⚠️ **Music Flamingo** - Optional re-ranking (disabled by default)

**Implementation Details**:
- Music Flamingo integration exists but is disabled by default
- Can be enabled with `use_music_flamingo=True`
- Computationally intensive, only used on top candidates
- ⚠️ **TODO**: Optimize Music Flamingo integration for production use

---

## Discover Weekly Features

### ✅ Completed Features

#### 1. **User History Management** (`user_history.py`)

- ✅ **Last-n interactions**: Tracks last N interactions per user (default: 50)
- ✅ **Weekly mixing**: 
  - If user has no interactions this week → use all historical interactions
  - If user has interactions this week → 50% historical + 50% this week's new interactions
- ✅ **Weighted interactions**: Supports play counts and other interaction weights
- ✅ **Weekly updates**: Models update every Monday (configurable)

#### 2. **Item-Based CF (Discover Weekly Style)**

- ✅ Precomputed item-item similarity matrix
- ✅ User-to-item indexing
- ✅ Weighted similarity (if play counts available)
- ✅ Top-k per seed song, then deduplicate (default: k=5)
- ✅ Weekly model updates

**How it works:**
1. Get user's last-n interactions (with weekly mixing)
2. For each seed song, find top-k most similar items (default: 20)
3. Aggregate scores across all seed songs
4. Deduplicate and return top N items

#### 3. **All Channels Support Last-N**

- ✅ **Item-Based CF**: Uses last-n interactions as seed songs
- ✅ **Item Content-Based**: Default uses last-n (`use_last_n=True`)
- ✅ **Prompt-Based**: Default uses last-n (`use_last_n=True`)
- ✅ **User-Based CF**: Uses user_id directly for recommendations
- ✅ **Two-Tower**: Uses last-n CLAP embeddings for user representation

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

#### 2. **Weekly Automatic Updates**

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
- **Prompt-Based**: Uses CLAP text embeddings (aligned with audio), defaults to last-n
- **User-Based CF**: Uses collaborative similarity, based on user_id
- **Two-Tower**: Uses CLAP audio embeddings, averages user's last-n tracks

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
│   ├── prompt_based.py          # Default use_last_n=True, CLAP text embeddings
│   ├── two_tower_recommender.py # CLAP audio embeddings
│   ├── din_ranker.py            # DIN for CTR prediction
│   └── user_based.py            # User-based CF
├── utils/
│   ├── clap_embeddings.py        # CLAP audio & text embeddings
│   ├── music_flamingo_quality.py # Music Flamingo integration
│   └── weekly_update.py         # Weekly update scheduler
├── weekly_update.py             # Executable update script
└── discover_weekly_example.py   # Complete example
```

---
