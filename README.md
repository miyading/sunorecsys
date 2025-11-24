# Suno Music Recommender System

A production-ready hybrid music recommender system for Suno-generated music, inspired by Spotify's personalized recommendations and modern social media platforms (Instagram, LittleRedBook).

## Architecture

The system implements a **four-stage hybrid architecture** for discover weekly music recommendation:

### ✅ Stage 1 (Recall): Candidate Retrieval
- ✅ **Channel 1: Item-based CF** - User-item interaction matrix based similarity (top-k per seed)
- ✅ **Channel 2: User-based CF** - Matrix factorization (ALS) based on user-item interactions
- ✅ **Channel 3: Two-tower model** - CLAP-based content retrieval (audio embeddings)

### ✅ Stage 2 (Coarse Ranking): Quality Filter
- ✅ **Channel 4: Quality filter** - Engagement-based quality scoring
- ⚠️ **TODO**: Integrate Meta Audiobox Aesthetics for better quality scoring

### ✅ Stage 3 (Fine Ranking): CTR Prediction
- ✅ **Channel 5: DIN with attention** - CTR prediction using user history (attention-based aggregation)
- ✅ **Channel 6: Prompt-based similarity** - CLAP text embeddings (aligned with audio space)

### ⚠️ Stage 4 (Re-ranking): Final Ranking
- ⚠️ **Music Flamingo** - Optional re-ranking (disabled by default, computationally intensive)

**Discover Weekly Features**:
- ✅ Automatic last-n interaction usage (default: 50 interactions)
- ✅ Weekly mixing: 50% historical + 50% this week's new interactions
- ✅ Weekly model updates (every Monday)
- ✅ Top-k per seed song (k=5) for Item-CF

## Key Design Highlights

### ✅ Completed Features

1. **Multi-Modal Alignment**: CLAP provides aligned text-audio embeddings, enabling direct similarity between prompts and audio tracks
2. **DIN for CTR Prediction**: Attention-based aggregation of user history for personalized ranking
3. **Hierarchical Aggregation Potential**: Same attention mechanism can aggregate tracks → artists
4. **AI Music Platform Specificity**: Leverages prompt-based creative intent as a unique signal

### ⚠️ Future Enhancements (TODOs)

1. **Meta Audiobox Aesthetics Integration** - Replace heuristic quality scoring with actual music quality models
2. **Music Flamingo Optimization** - Optimize Music Flamingo integration for production use (currently optional/disabled)
3. **Hierarchical Embedding Aggregation** - Implement artist-level aggregation from track embeddings using DIN-style attention

## Project Structure

```
sunorecsys/
├── data/              # Data processing and loading
│   └── user_history.py    # User history management
├── models/            # ML models and embeddings
├── recommenders/      # Recommendation algorithms
│   ├── hybrid.py          # Four-stage hybrid recommender
│   ├── item_cf.py         # Item-based CF (Stage 1)
│   ├── user_based.py      # User-based CF (Stage 1)
│   ├── two_tower_recommender.py  # CLAP-based retrieval (Stage 1)
│   ├── quality_filter.py  # Quality filtering (Stage 2)
│   ├── din_ranker.py      # DIN for CTR prediction (Stage 3)
│   └── prompt_based.py    # CLAP text embeddings (Stage 3)
├── evaluation/        # Evaluation metrics and testing
├── api/               # Production API service
├── config/            # Configuration files
└── utils/             # Utility functions
    ├── clap_embeddings.py      # CLAP audio & text embeddings
    └── music_flamingo_quality.py  # Music Flamingo integration
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

## Quick Start

### Step 1: Compute CLAP Embeddings

Compute CLAP audio embeddings for your songs:

```bash
python compute_clap_embeddings.py \
    --input data/songs.json \
    --output data/clap_embeddings.json
```

### Step 2: Train Two-Tower Model (Optional)

Train the two-tower model for CLAP-based retrieval:

```bash
python train_two_tower.py \
    --songs data/songs.json \
    --clap-embeddings data/clap_embeddings.json \
    --output models/two_tower.pt
```

### Step 3: Train/Fit the Recommender

Train the hybrid recommender on your processed data:

```bash
python run_recsys.py
```

This will:
1. Load songs and check CLAP embeddings
2. Train/fit the hybrid recommender (all stages)
3. Test recommendation scenarios
4. Show detailed output including stage, channel, and scores
5. Save the trained model

```

## Sample Output

When running `python run_recsys.py`, the system generates recommendations through a four-stage pipeline. Here's a sample output highlighting the key stages:

### Seed User
```
User: 000fefc6-d99e-41c9-b77e-f58b6689fc76 (has 3 songs)
📊 Using 16 last-n interactions for user 000fefc6-d99e-41c9-b77e-f58b6689fc76
```

### Stage 1: Recall - Candidate Retrieval

The system retrieves candidates from three channels:

**Channel 1: Item-Based CF** (Top 5 shown)
```
1. Wine Up Di Fire 🔥🔥
   Score: 0.5774
   Genre: Reggae
   From user history: [Solely "Suno"](https://suno.com/song/29a54268-b653-48c3-8b72-f35a8f4577b2)

2. Aeon
   Score: 0.5774
   Genre: retro electro cyberpunk
   From user history: [Upside Down Frown](https://suno.com/song/789f275d-58dc-46d3-8d46-7bbc327f9739)
```

**Channel 2: User-Based CF** (Top 5 shown)
```
1. Air before the Storm (https://suno.com/song/e48c0ee8-24c8-459b-8d91-469dbd6fc3a5)
   Score: 0.3248
   Genre: Electro-pop
   From similar user [8466026c-0b76-4cb0-8209-2564e8c0ff65]'s last-n

2. ひかりの岸辺 (https://suno.com/song/f245d20e-241d-4a5e-886a-cfeefc0e872a)
   Score: 0.3162
   Genre: Bouzouki-based Rebetiko
   From similar user [d361b865-bc5c-4239-9afa-0c810bdd4b6e]'s last-n
```

**Channel 3: Two-Tower CLAP** (Top 5 shown)
```
1. dialectic (accept it) (https://suno.com/song/e48c0ee8-24c8-459b-8d91-469dbd6fc3a5)
   Score: -0.4579
   Genre: post-indietronica
   From user history (average query): [The Fight Of Our Lives], [Last Door Left], [Upside Down Frown]
...
```

**Recall Summary**: 86 unique candidates retrieved from all channels

### Stage 2: Coarse Ranking - Quality Filter

Quality filter removes low-quality candidates:

```
📊 Quality Filter Statistics:
   Threshold: 0.3000
   Mean score: 0.2102
   Passed: 4/86 (4.7%)
   Filtered: 82/86 (95.3%)
✅ Coarse Ranking: 4 candidates after quality filtering
```

### Stage 3: Fine Ranking - CTR Prediction

Final candidates are scored using DIN (CTR prediction) and prompt-based similarity:

```
✅ Fine Ranking: 4 candidates scored with DIN (CTR prediction) + Prompt
```

### Final Recommendations

The system returns the top recommendations with detailed scores:

```
1. ビタミンガール！　果物と (https://suno.com/song/4a48a0c2-ad17-4963-9ddc-b506b268ecf8)
   Score: 0.3329
   Stage: Stage 1 (Recall)
   Primary Channel: Channel 2 (User-Based CF)
   Recall Channel Scores:
     - user_cf: 0.0503
   Fine Ranking Scores:
     - DIN (CTR Prediction): 0.3329
     - Prompt-based: 0.0000
   Genre: Synth-driven J-Pop Idol-kei

2. ひかりの岸辺 (https://suno.com/song/f245d20e-241d-4a5e-886a-cfeefc0e872a)
   Score: 0.3262
   Stage: Stage 1 (Recall)
   Primary Channel: Channel 2 (User-Based CF)
   Recall Channel Scores:
     - user_cf: 0.0949
   Fine Ranking Scores:
     - DIN (CTR Prediction): 0.3262
     - Prompt-based: 0.0000
   Genre: Bouzouki-based Rebetiko

3. Air Flows (https://suno.com/song/4914561d-5689-4dd5-ab6d-7262d0606fe2)
   Score: 0.3199
   Stage: Stage 1 (Recall)
   Primary Channel: Channel 1 (Item-Based CF)
   Recall Channel Scores:
     - item_cf: 0.1549
   Fine Ranking Scores:
     - DIN (CTR Prediction): 0.3199
     - Prompt-based: 0.0000
   Genre: orchestra
```

**Statistics**:
- Score range: [0.3197, 0.3329]
- Average score: 0.3247
- Recommendations by Stage: Stage 1 (Recall): 4 recommendations
- Top contributing recall channels: user_cf: 3, item_cf: 1

## Architecture Details

### Four-Stage Pipeline

```
User Query (user_id or song_ids)
    ↓
[Stage 1: Recall] - Candidate Retrieval
    ├─→ Item-based CF
    ├─→ User-based CF 
    └─→ Two-tower CLAP 
    weighted
    ↓
[Stage 2: Coarse Ranking] - Quality Filter
    └─→ Quality filter 
    ↓
[Stage 3: Fine Ranking] - CTR Prediction
    ├─→ DIN with attention - CTR prediction
    └─→ Prompt-based CLAP - User exploration
    weighted
    ↓
[Stage 4: Re-ranking] - Final Ranking (Optional)
    └─→ Music Flamingo (TODO)
    ↓
Final Recommendations
```

### Channel Details

**Stage 1 (Recall)**:
- **Item-based CF**: Item-item similarity, top-k per seed from user history
- **User-based CF**: User-user similarity, top-k per seed from user history
- **Two-tower**: Average of CLAP audio embeddings to represent user for user tower, CLAP audio embedding for item tower

**Stage 2 (Coarse Ranking)**:
- **Quality Filter**: Engagement-based scoring (plays, upvotes, comments)
- ⚠️ **TODO**: Meta Audiobox Aesthetics metrics integration

**Stage 3 (Fine Ranking)**:
- **DIN**: Attention-based aggregation of user history for CTR prediction
- **Prompt-based**: CLAP text embeddings (aligned with audio) for creative intent matching

**Stage 4 (Re-ranking)**:
- ⚠️ **TODO** Optional quality-based re-ranking

## Weekly Model Updates

```bash
# Run weekly update (every Monday)
python weekly_update.py

```

## Configuration

Default channel weights:
```python
HybridRecommender(
    # Stage 1 (Recall)
    item_cf_weight=0.30,      # Item-based CF
    user_cf_weight=0.30,      # User-based CF
    two_tower_weight=0.40,    # Two-tower CLAP
    
    # Stage 2 (Coarse Ranking)
    quality_threshold=0.3,    # Quality filter threshold
    
    # Stage 3 (Fine Ranking)
    din_weight=0.70,          # DIN CTR prediction
    prompt_weight=0.30,       # Prompt-based exploration
    
    # Stage 4 (Re-ranking)
    use_music_flamingo=False, # Music Flamingo (optional)
)
```


### ⚠️ TODO / Future Work

- ⚠️ **Meta Audiobox Aesthetics** - Enhance quality scoring
- ⚠️ **Music Flamingo for Re-ranking** - Through automatic captioning and LLM reasoning, optimize diversity and chaining in final playlist
- ⚠️ **Hierarchical Aggregation** - Artist-level embeddings from tracks (DIN-style attention)
- ⚠️ **Batch Processing**: Use batch recommendations for multiple users
- ⚠️ **Model Training** - Train on actual user interaction data, with weekly retraining
- ⚠️ **Monitoring**: Track recommendation quality and user engagement
- ⚠️ **A/B Testing**: Test different weight configurations


## Documentation

- **[DISCOVER_WEEKLY.md](DISCOVER_WEEKLY.md)** - Complete Discover Weekly implementation guide and system design summary
- **[CHANNEL_STRUCTURE.md](CHANNEL_STRUCTURE.md)** - Detailed channel architecture and design choices
- **[SIMULATED_INTERACTIONS.md](SIMULATED_INTERACTIONS.md)** - Data simulation documentation
- **[MUSIC_FLAMINGO_QUALITY.md](MUSIC_FLAMINGO_QUALITY.md)** - Music Flamingo integration guide
