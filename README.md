# Suno Music Recommender System

![Discover Weekly UI](../playlist_ui/Discover_Weekly_UI.png)

A production-ready hybrid music recommender system for Suno-generated music, inspired by Spotify's personalized recommendations and modern social media platforms (Instagram, LittleRedBook).

## UI Implementation

The `index.html` in `../playlist_ui/` provides an interactive 3D music visualizer that transforms album covers into orbiting particle systems. Built with Three.js, it features a central vinyl record player and allows users to interact with floating album art to play tracks or open external links. The implementation uses WebGL shaders for particle effects, raycasting for 3D object selection, and simplex noise-based animations.

## Recommender System Architecture

The system implements a **four-stage hybrid architecture** for discover weekly music recommendation:

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
- ⚠️ **TODO** Optional Music Flamingo quality-based re-ranking

## Key Design Highlights

### ✅ Completed Features

1. **Multi-Modal Alignment**: CLAP provides aligned text-audio embeddings, enabling direct similarity between prompts and audio tracks
2. **DIN for CTR Prediction**: Attention-based aggregation of user history for personalized ranking
3. **Hierarchical Aggregation Potential**: Same attention mechanism can aggregate tracks → artists
4. **AI Music Platform Specificity**: Leverages prompt-based creative intent as a unique signal

### ⚠️ Future Enhancements (TODOs)

1. **Meta Audiobox Aesthetics Integration** - Enhance quality scoring with actual music quality models
2. **Music Flamingo Optimization** - Through automatic captioning and LLM reasoning, optimize diversity and chaining in final playlist
3. **Hierarchical Embedding Aggregation** - Implement artist-level aggregation from track embeddings using DIN-style attention
4. **Batch Processing**: Use batch recommendations for multiple users
5. **Model Update** - Train on actual user interaction data, with weekly retraining
6. **Monitoring**: Track recommendation quality and user engagement
7. **A/B Testing**: Test different weight configurations


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

## Sample Output

When running `python run_recsys.py`, the system generates recommendations through a four-stage pipeline:

```text
================================================================================
🎵 Suno Recommendation System - Full Pipeline
================================================================================

[Step 3] User-based recommendations (using last-n interactions)
User: 000fefc6-d99e-41c9-b77e-f58b6689fc76 (has 3 songs)
📊 Using 16 last-n simulated interactions for user 000fefc6-d99e-41c9-b77e-f58b6689fc76

================================================================================
🔍 STAGE 1: RECALL - Candidate Retrieval
================================================================================

[Recall Channel 1] Item-Based CF...
  ✅ Retrieved 30 candidates from Item-Based CF

  📊 Top 5 Item-Based CF Recommendations:
     1. Wine Up Di Fire 🔥🔥
        Score: 0.5774
        Genre: Reggae
        From user history: [Solely "Suno"](https://suno.com/song/29a54268-b653-48c3-8b72-f35a8f4577b2)
     2. Aeon
        Score: 0.5774
        Genre: retro electro cyberpunk
        From user history: [Upside Down Frown](https://suno.com/song/789f275d-58dc-46d3-8d46-7bbc327f9739)
     3. Air Flows
        Score: 0.5164
        Genre: orchestra
        From user history: [End of the night](https://suno.com/song/8abce29f-432a-4935-addb-fce7835a1c25)
     4. Hum 💋-> Trumpet 📯
        Score: 0.5000
        Genre: unknown
        From user history: [How Much Wood? #TTChallenge](https://suno.com/song/0d48e1ea-121c-449b-ab61-59432be3fc1c)
     5. 「戦後の寝室 (After the War)」98-3, Our Bedroom After the War (Remix)
        Score: 0.5000
        Genre: 80s Japanese enka track with 8-bit chiptune
        From user history: [How Much Wood? #TTChallenge](https://suno.com/song/0d48e1ea-121c-449b-ab61-59432be3fc1c)

[Recall Channel 2] User-Based CF...
  ✅ Retrieved 30 candidates from User-Based CF

  📊 Top 5 User-Based CF Recommendations:
     1. Air before the Storm
        Score: 0.3248
        Genre: Electro-pop
        From similar user [8466026c-0b76-4cb0-8209-2564e8c0ff65]'s last-n
     2. ひかりの岸辺
        Score: 0.3162
        Genre: Bouzouki-based Rebetiko
        From similar user [d361b865-bc5c-4239-9afa-0c810bdd4b6e]'s last-n
     3. Make Round Thing
        Score: 0.3112
        Genre: tribal
        From similar user [e642bb49-03ea-4135-b461-fde312773129]'s last-n
     4. Tango with Mangoes #TTChallenge
        Score: 0.3112
        Genre: Wabbajack
        From similar user [2226f57a-3073-49ea-9381-d4edd39d4d24]'s last-n
     5. 醉打蔣門神
        Score: 0.3089
        Genre: heavy metal
        From similar user [2226f57a-3073-49ea-9381-d4edd39d4d24]'s last-n

[Recall Channel 3] Two-Tower Content Retrieval (CLAP-based)...
  ✅ Retrieved 30 candidates from Two-Tower

  📊 Top 5 Two-Tower Recommendations:
     1. dialectic (accept it)
        Score: -0.4579
        Genre: post-indietronica
        From user history (average query): [The Fight Of Our Lives], [Last Door Left], [Upside Down Frown]
     2. Ain't Got a Nickel Ain't Got a Dime
        Score: -0.5239
        Genre: up tempo Memphis soul
     3. ポートフォリオ
        Score: -0.5465
        Genre: J Hip Hop
     4. Golden Sunshine 윤아
        Score: -0.5571
        Genre: bossa nova
     5. Bossa Jazz A Cappella
        Score: -0.5636
        Genre: a cappella

📊 Recall Summary: 86 unique candidates retrieved

================================================================================
⚖️  STAGE 2: COARSE RANKING - Quality Filter
================================================================================
  ✅ Applied quality filter: 82 candidates filtered out

  📊 Quality Filter Statistics:
     Threshold: 0.3000
     Mean score: 0.2102
     Passed: 4/86 (4.7%)
     Filtered: 82/86 (95.3%)
  ✅ Coarse Ranking: 4 candidates after quality filtering

================================================================================
🎯 STAGE 3: FINE RANKING - CTR Prediction
================================================================================

[Fine Ranking Channel 5] DIN with Attention (CTR Prediction)...
[Fine Ranking Channel 6] Prompt-Based Similarity (User Exploration)...
  ✅ Fine Ranking: 4 candidates scored with DIN (CTR prediction) + Prompt

✅ Final Recommendations: 4 songs selected

Top Recommendations:

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

 4. Доторкнутися до любові    #Поезія_Птасі  #TTChallenge (https://suno.com/song/23342062-b394-431f-985f-24eebf84e958)
     Score: 0.3197
     Stage: Stage 1 (Recall)
     Primary Channel: Channel 2 (User-Based CF)
     Recall Channel Scores:
       - user_cf: 0.0500
     Fine Ranking Scores:
       - DIN (CTR Prediction): 0.3197
       - Prompt-based: 0.0000
     Genre: Hypnotic

[Step 4] Statistics
   Score range: [0.3197, 0.3329]
   Average score: 0.3247
   Recommendations by Stage:
     - Stage 1 (Recall): 4 recommendations
   Top contributing recall channels:
     - user_cf: 3 recommendations
     - item_cf: 1 recommendations
```

**Key Highlights**:
- **Seed User**: `000fefc6-d99e-41c9-b77e-f58b6689fc76` with 16 last-n interactions
- **Stage 1 (Recall)**: 86 candidates retrieved from 3 channels (Item CF, User CF, Two-Tower)
- **Stage 2 (Coarse Ranking)**: Quality filter reduces to 4 candidates (95.3% filtered)
- **Stage 3 (Fine Ranking)**: DIN CTR prediction + prompt-based scoring applied
- **Final Output**: Top 4 recommendations with detailed channel scores and metadata


## Architecture Details

### Four-Stage Pipeline

```
User Query (user_id or song_ids)
    |
    v
[Stage 1: Recall] - Candidate Retrieval
    |---> Item-based CF
    |---> User-based CF 
    |---> Two-tower CLAP 
    (weighted combination)
    |
    v
[Stage 2: Coarse Ranking] - Quality Filter
    |---> Quality filter 
    |
    v
[Stage 3: Fine Ranking] - CTR Prediction
    |---> DIN with attention - CTR prediction
    |---> Prompt-based CLAP - User exploration
    (weighted combination)
    |
    v
[Stage 4: Re-ranking] - Final Ranking (Optional)
    |---> Music Flamingo (TODO)
    |
    v
Final Recommendations
```

## Documentation

- **[DISCOVER_WEEKLY.md](DISCOVER_WEEKLY.md)** - Complete Discover Weekly implementation guide and system design summary
- **[CHANNEL_STRUCTURE.md](CHANNEL_STRUCTURE.md)** - Detailed channel architecture and design choices
- **[SIMULATED_INTERACTIONS.md](SIMULATED_INTERACTIONS.md)** - Data simulation documentation
- **[MUSIC_FLAMINGO_QUALITY.md](MUSIC_FLAMINGO_QUALITY.md)** - Music Flamingo integration guide
