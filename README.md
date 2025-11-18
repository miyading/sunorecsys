# Suno Music Recommender System

A production-ready hybrid music recommender system for Suno-generated music, inspired by Spotify's personalized recommendations and modern social media platforms (Instagram, LittleRedBook).

## Architecture

The system implements a hybrid recommendation approach with 5 channels (Discover Weekly style):

1. **Item-based Collaborative Filtering**: User-item interaction matrix based similarity (top-k per seed)
2. **Item Content-Based**: Unified content-based approach (embeddings + genre/metadata)
3. **Prompt-based Similarity**: Similarity based on generation prompts (unique to music generation models)
4. **User-based Collaborative Filtering**: Matrix factorization (ALS) based on user-item interactions
5. **Quality Filtering**: Engagement-based quality scores

**Discover Weekly Features**:
- Automatic last-n interaction usage (default: 50 interactions)
- Weekly mixing: 50% historical + 50% this week's new interactions
- Weekly model updates (every Monday)
- Top-k per seed song (k=5) for Item-CF

## Project Structure

```
sunorecsys/
├── data/              # Data processing and loading
├── models/            # ML models and embeddings
├── recommenders/      # Recommendation algorithms
├── evaluation/        # Evaluation metrics and testing
├── api/               # Production API service
├── config/            # Configuration files
└── utils/             # Utility functions
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

### Step 1: Preprocess Data

Process your song data to extract features:

```bash
python -m sunorecsys.data.preprocess --input all_songs.json --output data/processed
```

This will:
- Extract tags, genres, and metadata
- Process prompts and engagement metrics
- Save processed data to `data/processed/`

### Step 2: Train the Recommender

Train the hybrid recommender on your processed data:

```bash
python -m sunorecsys.recommenders.train \
    --songs data/processed \
    --playlists results.json \
    --output models/hybrid_recommender.pkl
```

You can adjust weights for different channels:
```bash
python -m sunorecsys.recommenders.train \
    --songs data/processed \
    --output models/hybrid_recommender.pkl \
    --item-weight 0.3 \
    --prompt-weight 0.25 \
    --genre-weight 0.15 \
    --user-weight 0.2 \
    --quality-weight 0.1
```

### Step 3: Use the Recommender

#### Python API

```python
from sunorecsys.recommenders.hybrid import HybridRecommender
import pandas as pd

# Load model
recommender = HybridRecommender.load("models/hybrid_recommender.pkl")

# Get recommendations based on seed songs
recommendations = recommender.recommend(
    song_ids=["song-id-1", "song-id-2"],
    n=10
)

# Get similar songs
similar = recommender.get_similar_songs("song-id-1", n=5)

# User-based recommendations
recommendations = recommender.recommend(
    user_id="user-id-1",
    n=20
)
```

#### REST API

Start the API server:

```bash
uvicorn sunorecsys.api.main:app --host 0.0.0.0 --port 8000
```

Then use the API:

```bash
# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "song_ids": ["song-id-1", "song-id-2"],
    "n": 10
  }'

# Get similar songs
curl "http://localhost:8000/similar/song-id-1?n=5"

# Health check
curl "http://localhost:8000/health"
```

### Step 4: Test the System

Run the complete test script:

```bash
python run_recsys.py
```

This will:
1. Load songs and check CLAP embeddings
2. Train/fit the hybrid recommender
3. Test 3 recommendation scenarios:
   - Seed song-based recommendations
   - Similar songs
   - User-based recommendations
4. Show detailed output including channel scores and metadata
5. Save the trained model

### Step 5: Evaluate (Optional)

Evaluate your recommender on test data:

```python
from sunorecsys.evaluation.metrics import evaluate_recommender
from sunorecsys.recommenders.hybrid import HybridRecommender

recommender = HybridRecommender.load("models/hybrid_recommender.pkl")

# Prepare test data
test_data = [
    {
        "song_ids": ["seed-1", "seed-2"],
        "relevant": {"song-1", "song-2", "song-3"}  # Ground truth
    },
    # ... more test cases
]

# Evaluate
results = evaluate_recommender(
    recommender,
    test_data,
    k_values=[5, 10, 20]
)

print(results)
```

## Weekly Model Updates

```bash
# Run weekly update (every Monday)
python weekly_update.py

# Or schedule with cron:
# 0 0 * * 1 cd /path/to/sunorecsys && python weekly_update.py
```

## Configuration

See `config/` directory for configuration files. The system supports:
- Weight tuning for hybrid channels
- Embedding model selection
- Quality thresholds
- Similarity metrics

## Architecture Overview

The system uses 5 recommendation channels:

1. **Item-based Similarity** (25%): Embedding-based similarity using tags, prompts, and metadata
2. **Prompt-based Similarity** (20%): Similarity based on generation prompts
3. **Genre/Metadata-based** (20%): Recommendations based on genre and tag patterns
4. **User-based CF** (25%): Collaborative filtering using playlist co-occurrence
5. **Quality Filtering** (10%): Filters low-quality songs using engagement metrics

The hybrid recommender combines all channels with weighted scores.

For detailed architecture and design choices, see [CHANNEL_STRUCTURE.md](CHANNEL_STRUCTURE.md).

## Production Considerations

1. **Caching**: The API can be extended with Redis caching for frequent queries
2. **Batch Processing**: Use batch recommendations for multiple users
3. **Model Updates**: Retrain periodically as new songs are added
4. **Monitoring**: Track recommendation quality and user engagement
5. **A/B Testing**: Test different weight configurations

## Next Steps

- Integrate Meta Audiobox Aesthetics for better quality scoring
- Add audio embeddings for content-based recommendations
- Implement real-time learning from user feedback
- Add diversity constraints to recommendations
- Build a web UI for interactive recommendations

## Documentation

- [CHANNEL_STRUCTURE.md](CHANNEL_STRUCTURE.md) - Detailed channel architecture and design choices
- [DISCOVER_WEEKLY.md](DISCOVER_WEEKLY.md) - Discover Weekly implementation guide
- [SIMULATED_INTERACTIONS.md](SIMULATED_INTERACTIONS.md) - Data simulation documentation
- [CLAP_EMBEDDINGS_USAGE.md](CLAP_EMBEDDINGS_USAGE.md) - CLAP embeddings guide
