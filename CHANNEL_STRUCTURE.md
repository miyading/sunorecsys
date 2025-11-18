# Recommendation Channel Structure

## Overview

The hybrid recommender system uses a clearer channel structure that separates collaborative filtering from content-based methods. This document details the architecture, design choices, and implementation details for each channel.

## System Architecture

The Suno Music Recommender System is a production-ready hybrid recommendation engine that combines multiple recommendation strategies to provide personalized music recommendations. The system is designed to be modular, extensible, and suitable for daily production use.

### Core Components

1. **Data Processing** (`sunorecsys/data/`): Extract and normalize features from raw song data
2. **Embedding Utilities** (`sunorecsys/utils/embeddings.py`): Generate text embeddings for prompts and tags
3. **Recommendation Channels**: Five specialized recommendation strategies
4. **Hybrid Recommender**: Combines all channels with weighted scores
5. **Evaluation Framework**: Metrics for offline evaluation
6. **API Service**: Production-ready REST API

### Data Flow

```
Raw Songs (JSON)
    ↓
[Data Preprocessing]
    ↓
Processed Songs (DataFrame)
    ↓
[Feature Extraction]
    ├─→ Tags
    ├─→ Prompts
    ├─→ Metadata
    └─→ Engagement Metrics
    ↓
[Embedding Generation]
    ├─→ Tag Embeddings
    └─→ Prompt Embeddings
    ↓
[Model Training]
    ├─→ Item-Based Model
    ├─→ Prompt-Based Model
    ├─→ Genre-Based Model
    ├─→ User-Based CF Model
    └─→ Quality Filter
    ↓
[Hybrid Recommender]
    ↓
Recommendations
```

---

## Channel Architecture

### Channel 1: Item-Based Collaborative Filtering (`ItemBasedCFRecommender`)
**Type**: Collaborative Filtering  
**Method**: User-item interaction matrix based similarity

**How it works**:
- Builds a user-item interaction matrix (users x songs)
- Computes item-item similarity using cosine similarity on item vectors
- Two songs are similar if the same group of users have interacted with both
- Uses the user-item interaction matrix directly (no matrix factorization)

**Key Features**:
- Based on actual user behavior patterns
- Captures "people who liked X also liked Y" relationships
- Uses cosine similarity on user interaction vectors
- Top-k per seed song (Discover Weekly style, default: k=5)
- Supports weighted similarity if play counts available

**When to use**:
- When you have user interaction data (playlists, likes, listens)
- For discovering songs that co-occur in user collections
- Good for cold-start items (if they appear in playlists)

**Data Required**:
- User-item interaction matrix (built from playlists or user listening history)

**Design Choices**:
- **User-Item Interaction Matrix**: Captures "people who liked X also liked Y" relationships
- **Item-Item Similarity via Cosine**: Fast computation with sparse matrices, interpretable
- **Weighted Similarity**: If play counts available, uses weighted cosine similarity
- **Top-K Per Seed Song**: For each seed song, find top-k most similar (default: k=5), then deduplicate
- **Precomputed Similarity Matrix**: Computed once during training, updated weekly

**Assumptions**:
- User co-occurrence = similarity (valid for well-curated playlists)
- Sparse matrix handling is sufficient
- Weekly similarity updates are sufficient

---

### Channel 2: Item Content-Based (`ItemContentBasedRecommender`)
**Type**: Content-Based  
**Method**: Unified approach combining embeddings, genre, tags, and popularity

**How it works**:
- Generates embeddings for tags and prompts
- Combines with explicit genre/tag matching
- Scores based on:
  - Embedding similarity (60% weight)
  - Genre matching (20% weight)
  - Tag matching (15% weight)
  - Popularity matching (5% weight)
- Uses Annoy index for fast similarity search
- Supports last-n interactions from user history

**Key Features**:
- Content-based (doesn't require user interactions)
- Captures both semantic and explicit feature similarity
- Fast retrieval with Annoy index
- Default uses last-n interactions for Discover Weekly style

**When to use**:
- For content-based recommendations
- When user interaction data is sparse
- For discovering musically similar songs
- When you want genre/tag-aware recommendations

**Data Required**:
- Song tags, prompts, genres, and metadata
- Optional: User listening history for preference modeling

**Design Choices**:
- **Unified Content-Based Approach**: Combines embeddings (60%), genre (20%), tags (15%), popularity (5%)
- **Multi-Modal Embedding**: Combines tag embeddings and prompt embeddings
- **Genre and Tag Matching**: Explicit matching on genre and tags from user history
- **Last-N Interactions Support**: Default to using last-n interactions from user history

**Assumptions**:
- Text embeddings capture musical similarity (strong for generated music)
- Genre consistency: users prefer consistent genres
- Last-n interactions are representative of user preferences

---

### Channel 3: Prompt-Based Similarity (`PromptBasedRecommender`)
**Type**: Content-Based (Special case of item similarity)  
**Method**: Prompt embedding similarity

**How it works**:
- Embeds generation prompts using sentence transformers
- Computes similarity between prompts
- Assumes: similar prompts → similar-sounding music

**Key Features**:
- Unique to AI-generated music
- Leverages the direct relationship between prompts and output
- Fast and effective for prompt-based discovery
- Supports last-n interactions from user history

**When to use**:
- For discovering songs with similar creative intent
- When prompts are detailed and descriptive
- For exploring variations of a musical concept

**Data Required**:
- Song generation prompts

**Design Choices**:
- **Full Prompt Embedding**: Embeds entire prompt text (including lyrics, structure markers)
- **Average Embedding for Multiple Seeds**: Averages embeddings of seed songs to create query vector
- **Unique to Generation Models**: Treats prompt similarity as a distinct signal

**Assumptions**:
- Similar prompts produce similar-sounding music (strong for well-trained models)
- Prompts contain sufficient information for similarity
- Text embeddings are sufficient (no audio analysis required initially)

---

### Channel 4: User-Based Collaborative Filtering (`UserBasedRecommender`)
**Type**: Collaborative Filtering  
**Method**: Matrix factorization (ALS) with user-item matrix

**How it works**:
- Builds user-item interaction matrix
- Factorizes into user and item latent factors
- For virtual users (seed songs): computes user factors by averaging item factors
- Recommends based on user similarity

**Key Features**:
- Captures user preferences in latent space
- Handles virtual users (hypothetical users from seed songs)
- Uses implicit feedback (binary interactions)

**When to use**:
- When you have rich user interaction data
- For personalized recommendations
- For discovering songs liked by similar users

**Data Required**:
- User-item interaction matrix

**Design Choices**:
- **Playlist-as-User Simulation**: Treats playlists as "users" for CF
- **Implicit Feedback (Binary)**: Binary interaction (song in playlist = 1, else 0)
- **Alternating Least Squares (ALS)**: Matrix factorization with ALS algorithm
  - `factors=50`: Latent dimension
  - `regularization=0.1`: Prevents overfitting
  - `iterations=15`: Convergence trade-off

**Assumptions**:
- Playlists are well-curated
- Songs in same playlist are similar
- ALS handles sparse data well
- Similar users have similar preferences

---

### Channel 5: Quality Filtering (`QualityFilter`)
**Type**: Filter  
**Method**: Engagement-based quality scoring

**How it works**:
- Computes quality scores based on:
  - Engagement rate (upvotes + comments) / plays
  - Popularity (log of plays)
  - Upvote ratio (upvotes / plays)
- Filters out songs below quality threshold
- Boosts scores of high-quality songs

**Key Features**:
- Ensures only high-quality songs are recommended
- Configurable quality threshold
- Works as a final gate after channel combination

**When to use**:
- Always enabled by default
- For filtering low-quality content
- For boosting high-quality recommendations

**Data Required**:
- Song engagement metrics (plays, upvotes, comments)

**Design Choices**:
- **Engagement-Based Heuristic**: Combines engagement rate (40%), popularity (30%), upvote ratio (30%)
- **Threshold-Based Filtering**: Filters songs below quality threshold (default: 0.3)
- **Score Boosting**: Boosts recommendation scores by quality (0.7 + 0.3 * quality_score)

**Assumptions**:
- High engagement indicates quality (generally true, but popularity bias exists)
- Normalized quality scores are comparable
- Quality doesn't change over time (reasonable for most songs)

---

## User-Item Interaction Matrix

**The system uses user-item interaction matrices!**

### Where it's used:
1. **Item-Based CF** (`item_cf.py`):
   - Builds `user_item_matrix` (users x songs)
   - Uses it to compute item-item cosine similarity
   - Matrix shape: `(num_users, num_songs)`

2. **User-Based CF** (`user_based.py`):
   - Builds `user_item_matrix` (users x songs)
   - Factorizes it using ALS: `user_item_matrix ≈ user_factors @ item_factors.T`
   - Matrix shape: `(num_users, num_songs)`

### How it's built:

**Current Implementation (Simplified Approach):**
- **From playlists**: Each playlist = a "user", songs in playlist = interactions
  - This treats each playlist as a separate user entity
  - Works when you don't have playlist creator information
  - Each playlist represents a distinct taste profile
- **From user_id**: Each user's songs = interactions (simulated playlists)
  - Groups songs by the `user_id` field in the songs DataFrame
  - Each user's collection becomes their interaction vector
- **Binary interactions**: Song in playlist/user collection = 1, else 0

**Alternative: Using Actual Playlist Creators (Recommended for Production):**
- **From playlists with creators**: Use `user_handle` or `user_id` from playlist metadata
  - Aggregates all playlists created by the same user
  - More accurate user profiles (users can have multiple playlists)
  - Better for personalized recommendations

**When Playlists Have Multiple Creators:**
- **Option 1**: Use the primary creator (first creator, owner)
- **Option 2**: Split interactions proportionally among creators
- **Option 3**: Create separate "collaborative playlist" entities
- **Current code uses Option 1** (treats playlist as single-user entity)

### Matrix Properties:
- **Sparse**: Most entries are 0 (users don't interact with most songs)
- **Binary**: Values are 0 or 1 (implicit feedback)
- **Asymmetric**: Different users have different interaction patterns

---

## Hybrid Combination Strategy

### Design Choices

**1. Weighted Linear Combination**
- **Choice**: Simple weighted sum of channel scores
- **Rationale**: Interpretable and tunable, fast computation, easy to A/B test
- **Limitation**: Assumes channels are independent (may not be true)

**2. Score Normalization**
- **Choice**: Normalize scores within each channel before combination
- **Rationale**: Prevents one channel from dominating, makes weights comparable

**3. Quality Filter Integration**
- **Choice**: Apply quality filter after combination
- **Rationale**: Removes low-quality songs regardless of recommendation score, acts as a final gate

### Default Weights

```python
HybridRecommender(
    item_cf_weight=0.25,         # Item-based CF
    item_content_weight=0.30,     # Item content-based (embeddings + genre/metadata)
    prompt_weight=0.20,          # Prompt-based
    user_weight=0.20,           # User-based CF
    quality_weight=0.05,         # Quality filter
    use_last_n=True,             # Default: use last-n interactions
)
```

---

## Channel Comparison

| Channel | Type | Data Source | Speed | Interpretability |
|---------|------|-------------|-------|------------------|
| Item-Based CF | Collaborative | User interactions | Fast | Medium |
| Item Content-Based | Content | Tags/Prompts/Genre | Very Fast | Medium |
| Prompt-Based | Content | Prompts | Very Fast | Medium |
| User-Based CF | Collaborative | User interactions | Medium | Low |
| Quality Filter | Filter | Engagement metrics | Very Fast | High |

---

## Discover Weekly Integration

### Design Choices

**1. User History Management**
- **Choice**: Track last-n interactions per user (default: 50) with temporal information
- **Rationale**: Enables Discover Weekly-style recommendations, balances recency with historical patterns

**2. Weekly Mixing Strategy**
- **Choice**: If user has interactions this week: 50% historical + 50% this week's new
- **Rationale**: Balances long-term preferences with recent activity, mimics Spotify's approach

**3. Automatic Last-N Usage**
- **Choice**: HybridRecommender automatically uses last-n interactions when user_id provided
- **Rationale**: Simplifies API, ensures consistent behavior across channels

**4. Weekly Model Updates**
- **Choice**: Update item-item similarity and all models every Monday
- **Rationale**: Incorporates new songs and interactions, balances freshness with computational cost

### Assumptions

- Weekly updates are optimal (matches user expectations)
- Last-n interactions capture user preferences (recent interactions are most predictive)
- 50/50 historical/new mixing is optimal (balances stability with freshness)

---

## Production Considerations

### Scalability

1. **Vector Search**: Uses Annoy for fast similarity search (O(log n))
2. **Sparse Matrices**: User-based CF uses sparse matrices for memory efficiency
3. **Batch Processing**: Supports batch recommendations
4. **Caching**: Can add Redis caching for frequent queries

### Performance

1. **Embedding Caching**: Embeddings are precomputed and stored
2. **Index Building**: Annoy indices are built once and saved
3. **Lazy Loading**: Models can be loaded on-demand

### Monitoring

1. **Metrics**: Evaluation framework for offline metrics
2. **Logging**: Structured logging for API requests
3. **Health Checks**: Built-in health check endpoints

### Extensibility

1. **New Channels**: Easy to add new recommendation channels
2. **Custom Embeddings**: Can swap embedding models
3. **Quality Models**: Can integrate external quality scoring models
4. **A/B Testing**: Support for multiple model configurations

---

## Future Enhancements

### Short-term
1. **Audio Embeddings**: Add audio-based similarity using CLAP embeddings (already implemented)
2. **Real-time Learning**: Update models from user feedback
3. **Diversity Constraints**: Ensure recommendation diversity
4. **Cold Start**: Better handling for new users/songs

### Medium-term
1. **Meta Audiobox Integration**: Use actual music quality models
2. **Deep Learning**: Neural collaborative filtering
3. **Multi-armed Bandits**: Online learning for weight tuning
4. **Explainability**: Provide explanations for recommendations

### Long-term
1. **Graph Neural Networks**: Model user-song relationships as graph
2. **Transformer Models**: Use transformer-based recommenders
3. **Multi-modal**: Combine audio, text, and visual features
4. **Personalization**: Per-user model fine-tuning

---

## Technical Stack

- **Python 3.8+**
- **NumPy/Pandas**: Data processing
- **Sentence Transformers**: Text embeddings
- **Annoy**: Fast similarity search
- **Implicit**: Collaborative filtering
- **FastAPI**: API framework
- **Scikit-learn**: ML utilities
- **CLAP**: Audio embeddings

---

## Usage Example

```python
from sunorecsys.recommenders.hybrid import HybridRecommender

recommender = HybridRecommender(
    item_cf_weight=0.30,      # Emphasize item-based CF
    item_content_weight=0.25,
    prompt_weight=0.20,
    user_weight=0.20,
    quality_weight=0.05,
    use_user_cf=True,         # Enable user-based CF
)

recommender.fit(songs_df, playlists=playlists)

# Get recommendations with detailed channel breakdown
recommendations = recommender.recommend(
    song_ids=["seed-song-1", "seed-song-2"],
    n=20,
    return_details=True  # See which channels contributed
)

# Each recommendation includes:
# - song_id, title, score
# - channel_scores: breakdown by channel
# - details: channel-specific debugging info
```

---

## Key Differences from Previous Structure

### Current Structure:
- Channel 1: Item-based CF - **collaborative** (user-item interactions)
- Channel 2: Item content-based - **content-based** (embeddings + genre/metadata unified)
- Channel 3: Prompt-based - **content-based** (prompt similarity)
- Channel 4: User-based CF - **collaborative** (matrix factorization)
- Channel 5: Quality Filter - **filter** (quality filtering)

**Key Features:**
- Clear separation between collaborative (1, 4) and content-based (2, 3)
- Unified content-based channel avoids redundancy
- All channels support Discover Weekly style (last-n interactions)
- Quality filter ensures recommendation quality

---

## Virtual User Implementation

The user-based CF channel properly handles virtual users (hypothetical users created from seed songs):

1. **Create virtual user vector**: Binary vector indicating which seed songs the user "liked"
2. **Compute user factors**: Average the item factors of seed songs
3. **Score all items**: `user_factors @ item_factors.T`
4. **Recommend top items**: Exclude seed songs, return top-N

This creates a proper hypothetical user profile and uses collaborative filtering to find songs that similar users (those who liked the seed songs) also liked.

---

## Summary of Key Assumptions

| Channel | Key Assumption | Validity | Risk Level |
|---------|---------------|----------|-----------|
| Item-Based CF | User co-occurrence = similarity | High | Medium |
| Item Content-Based | Text embeddings + genre match = similarity | Medium | Medium |
| Prompt-Based | Prompt similarity = musical similarity | High | Low |
| User-Based CF | Similar users like similar items | Medium | Medium |
| Quality Filter | Engagement = quality | Medium | High |
