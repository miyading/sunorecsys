# Recommendation Channel Structure

This document details the architecture, design choices, and implementation details for each recommendation channel. For system overview and architecture, see [README.md](../README.md).

---

## Channel Architecture by Stage

The system uses a four-stage pipeline. Channels are organized by their stage in the recommendation pipeline.

---

## Stage 1: Recall - Candidate Retrieval

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

### Channel 2: User-Based Collaborative Filtering (`UserBasedRecommender`)
**Type**: Collaborative Filtering  
**Method**: User-user similarity via cosine similarity on user-item matrix

**How it works**:
- Builds user-item interaction matrix (users x songs)
- Computes user-user similarity using cosine similarity on user vectors
- Two users are similar if they have interacted with the same group of songs
- For each seed song: creates virtual user vector, finds similar users, aggregates their liked items
- Discover Weekly style: finds top-k similar users, gets top-k items from each user's last-n interactions

**Key Features**:
- Captures "users who liked X also liked Y" relationships
- Handles virtual users (hypothetical users created from seed songs)
- Uses implicit feedback (binary interactions)
- Supports last-n interactions per similar user (temporal ordering)

**Data Required**:
- User-item interaction matrix

**Design Choices**:
- **Playlist-as-User Simulation**: Treats playlists as "users" for CF
- **Implicit Feedback (Binary)**: Binary interaction (song in playlist = 1, else 0)
- **User-User Cosine Similarity**: Direct cosine similarity on user interaction vectors (no matrix factorization)
- **Top-K Similar Users**: Finds top-k most similar users (default: k=20)
- **Top-K Per Similar User**: Gets top-k items from each similar user's last-n interactions (default: k=5)
- **Last-N Per User**: Uses last-n interacted items per similar user (default: n=50) for temporal ordering

**Assumptions**:
- Playlists and last-n interactions are representative of user preferences

---

### Channel 3: Two-Tower Content Retrieval (`TwoTowerRecommender`)
**Type**: Content-Based  
**Method**: CLAP audio embeddings with two-tower neural network

**How it works**:
- **User Tower**: Averages CLAP audio embeddings of user's last-n interacted items to represent user preference
- **Item Tower**: Uses CLAP audio embeddings directly for each song
- **Retrieval**: Computes cosine similarity between user embedding and all item embeddings
- Returns top-k most similar items based on audio content

**Key Features**:
- Content-based retrieval using audio embeddings
- Leverages CLAP's aligned text-audio embedding space
- User representation from historical interactions (last-n)
- Fast retrieval with precomputed embeddings

**Data Required**:
- CLAP audio embeddings for all songs
- User interaction history (for user tower)
- Trained two-tower model checkpoint

**Design Choices**:
- **CLAP Audio Embeddings**: Uses precomputed CLAP embeddings for items
- **Average User Representation**: Simple averaging of last-n item embeddings (no attention)
- **Neural Two-Tower Model**: Trained model that can encode items with metadata
- **Cosine Similarity**: Fast retrieval using embedding similarity

**Assumptions**:
- Audio embeddings capture musical similarity
- Averaging last-n embeddings represents user preference
- CLAP embeddings are sufficient for content-based retrieval

---

### Channel 4: Prompt-Based Similarity (`PromptBasedRecommender`)
**Type**: Content-Based  
**Method**: CLAP text embeddings (aligned with audio) for creative intent matching

**How it works**:
- Uses CLAP text embeddings of generation prompts (aligned with audio embeddings)
- Takes seed songs from user's **listening history** (songs created by other artists, not user's own creations)
- Computes similarity between prompts of seed songs from listening history and prompts of candidate songs in the catalog
- Averages user's last-n prompt embeddings from listened songs to create query vector
- Retrieves songs with similar creative intent (prompt similarity) from the catalog using Annoy index

**Why Recall Stage**:
In this system, seed songs come from the user's simulated listening history—these are songs that other artists created, not the user's own creations. The prompt-based channel finds which songs in the catalog have generation prompts similar to the prompts of songs the user has listened to. This is fundamentally a candidate retrieval task: given a user's listening history, find songs in the catalog whose creative intent (captured by prompts) matches the creative intent of songs they've consumed. This is different from adding diversity to final recommendations (which would be a ranking/exploration signal). If seed songs were instead the user's own created songs (their generation prompts), then prompt-based similarity could serve as a diversity signal in fine ranking, similar to how ads are directly inserted (直接插入) in industry to occasionally show users content similar to what they create. However, with listening history as seeds, it's a retrieval task.

**Key Features**:
- Unique to AI-generated music platforms
- Leverages aligned text-audio embedding space (CLAP)
- Captures creative intent through prompt similarity
- Fast retrieval using precomputed CLAP text embeddings with Annoy index

**Data Required**:
- CLAP text embeddings for song generation prompts
- User interaction history (for last-n prompts from listened songs)

**Design Choices**:
- **CLAP Text Embeddings**: Uses CLAP's text encoder (aligned with audio)
- **Full Prompt Embedding**: Embeds entire prompt text (including lyrics, structure markers)
- **Average Embedding for Multiple Seeds**: Averages embeddings of seed songs' prompts to create query vector
- **Aligned Embedding Space**: Text and audio embeddings in same space enable unified similarity
- **Recall Stage Placement**: Positioned in recall because it retrieves candidates from catalog based on prompt similarity to listened songs

**Assumptions**:
- Similar prompts produce similar-sounding music (strong for well-trained models)
- Prompts contain sufficient information for similarity
- CLAP's aligned text-audio space enables effective prompt-based retrieval
- User's listening history reflects their prompt preferences (i.e., if they listened to songs with certain prompts, they may like other songs with similar prompts)

---

## Stage 2: Coarse Ranking - Quality Filter

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

## Stage 3: Fine Ranking - CTR Prediction

### Channel 6: DIN with Attention (`DINRanker`)
**Type**: Deep Learning / CTR Prediction  
**Method**: Attention-based aggregation of user history for CTR prediction

**How it works**:
- Uses Deep Interest Network (DIN) architecture
- **Attention Mechanism**: Computes relevance scores between candidate track and each historical track
- **User Representation**: Attention-weighted aggregation of user's historical track embeddings
- **CTR Prediction**: Predicts click-through rate for candidate tracks using user representation and candidate embedding
- Scores candidates based on predicted CTR

**Key Features**:
- Personalized ranking based on user history
- Attention mechanism learns which historical tracks are most relevant
- Uses CLAP embeddings for track representation
- Trained on user interaction data (positive and negative samples)

**Data Required**:
- CLAP embeddings for tracks
- User interaction history (for training and inference)
- Trained DIN model checkpoint

**Design Choices**:
- **Attention Aggregator**: Learns relevance between historical and candidate tracks
- **CLAP Embeddings**: Uses same embedding space as two-tower and prompt-based channels
- **Binary Classification**: Predicts CTR (click probability) for ranking
- **User History**: Uses last-n interactions with temporal ordering

**Assumptions**:
- Attention mechanism captures relevant historical patterns
- CLAP embeddings are sufficient for CTR prediction
- User's recent interactions are predictive of future clicks

---

## User-Item Interaction Matrix

**The system uses user-item interaction matrices for collaborative filtering channels.**

### Where it's used:
1. **Item-Based CF** (`item_cf.py`):
   - Builds `user_item_matrix` (users x songs)
   - Uses it to compute item-item cosine similarity
   - Matrix shape: `(num_users, num_songs)`

2. **User-Based CF** (`user_based.py`):
   - Builds `user_item_matrix` (users x songs)
   - Computes user-user similarity: `cosine_similarity(user_item_matrix)`
   - Matrix shape: `(num_users, num_songs)`

### How it's built:

**Current Implementation:**
- **From playlists**: Each playlist = a "user", songs in playlist = interactions
  - Treats each playlist as a separate user entity
  - Works when you don't have playlist creator information
  - Each playlist represents a distinct taste profile
- **From user_id**: Each user's songs = interactions (simulated playlists)
  - Groups songs by the `user_id` field in the songs DataFrame
  - Each user's collection becomes their interaction vector
- **With simulated interactions**: Can augment with simulated interactions for denser matrix
- **Binary interactions**: Song in playlist/user collection = 1, else 0

---

## Hybrid Combination Strategy

### Design Choices

**1. Weighted Linear Combination**
- **Choice**: Simple weighted sum of channel scores within each stage
- **Rationale**: Interpretable and tunable, fast computation, easy to A/B test
- **Limitation**: Assumes channels are independent (may not be true)

**2. Score Normalization**
- **Choice**: Normalize scores within each channel before combination
- **Rationale**: Prevents one channel from dominating, makes weights comparable

**3. Stage-wise Processing**
- **Stage 1 (Recall)**: Combines Item CF, User CF, Two-Tower (audio), and Prompt-Based (text) scores (weighted sum)
- **Stage 2 (Coarse Ranking)**: Quality filter removes low-quality candidates
- **Stage 3 (Fine Ranking)**: DIN CTR prediction for personalized ranking

### Default Weights

```python
HybridRecommender(
    # Stage 1 (Recall)
    item_cf_weight=0.25,        # Channel 1: Item-based CF
    user_cf_weight=0.25,        # Channel 2: User-based CF
    two_tower_weight=0.35,      # Channel 3: Two-tower content retrieval (audio)
    prompt_weight=0.15,          # Channel 4: Prompt-based similarity (text/creative intent)
    
    # Stage 2 (Coarse Ranking)
    quality_threshold=0.3,       # Channel 5: Quality filter (no weight, just filtering)
    
    # Stage 3 (Fine Ranking)
    din_weight=1.0,              # Channel 6: DIN CTR prediction (only channel in this stage)
    
    use_last_n=True,             # Default: use last-n interactions
)
```

---

## Virtual User Implementation

The user-based CF channel handles virtual users (hypothetical users created from seed songs):

1. **Create virtual user vector**: Binary vector indicating which seed songs the user "liked"
2. **Compute user similarity**: Cosine similarity between virtual user vector and all users in user-item matrix
3. **Find similar users**: Get top-k most similar users to the virtual user
4. **Aggregate recommendations**: For each similar user, get top-k items from their last-n interactions
5. **Score and rank**: Score = Σ similarity(user_virtual, user_j) × like(user_j, item) for all similar users j
6. **Recommend top items**: Exclude seed songs, return top-N

This creates a proper hypothetical user profile and uses collaborative filtering to find songs that similar users (those who liked the seed songs) also liked.
