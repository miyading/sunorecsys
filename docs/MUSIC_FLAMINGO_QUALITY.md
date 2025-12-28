# Music Flamingo Quality Filtering Guide

This guide explains how to use NVIDIA's Music Flamingo model for quality filtering in the Suno RecSys.

## Overview

Music Flamingo is a state-of-the-art Large Audio-Language Model (LALM) designed for music understanding. It can analyze audio tracks and provide multi-dimensional quality scores that are perfect for filtering low-quality recommendations.

## Installation

First, install the required dependencies:

```bash
# Install transformers with Music Flamingo support
pip install --upgrade pip
pip install --upgrade git+https://github.com/huggingface/transformers accelerate

# Install other dependencies if needed
pip install torch torchaudio numpy tqdm
```

## Quick Start

### 1. Compute Quality Scores for Songs

Use the provided script to compute quality scores:

```bash
python compute_music_flamingo_quality.py \
    --input sunorecsys/data/curl/all_songs.json \
    --output data/music_flamingo_scores.json \
    --cache-dir data/music_flamingo_scores \
    --device cuda  # or 'cpu' if no GPU
```

### 2. Use Precomputed Scores in Quality Filter

```python
from sunorecsys.recommenders.quality_filter import QualityFilter
import pandas as pd

# Load your songs
songs_df = pd.read_parquet("data/processed/songs.parquet")

# Initialize quality filter with precomputed Music Flamingo scores
quality_filter = QualityFilter(
    quality_threshold=0.4,  # Filter songs below this score
    quality_scores_file="data/music_flamingo_scores.json",
)

# Fit the filter
quality_filter.fit(songs_df)

# Use in recommendations
filtered_songs = quality_filter.filter(['song-1', 'song-2', 'song-3'])
```

## Quality Scoring Dimensions

Music Flamingo scores tracks on multiple dimensions:

1. **Content Usefulness** (0.0-1.0): How useful/meaningful is the content?
2. **Production Quality** (0.0-1.0): Technical production quality
3. **Content Enjoyment** (0.0-1.0): How enjoyable is the content?
4. **Production Complexity** (0.0-1.0): Complexity of production
5. **Overall Recommendation Score** (0.0-1.0): Should this be recommended?

The overall quality score is a weighted combination of these dimensions:
- Content Usefulness: 15%
- Production Quality: 30%
- Content Enjoyment: 25%
- Production Complexity: 10%
- Overall Recommendation: 20%

## Prompt Design

The quality scoring prompt is designed to get structured output:

```python
QUALITY_PROMPT_TEMPLATE = """Describe this Suno AI generated track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.

Then score the track on four dimensions [0,1]: 
1. Content Usefulness: How useful/meaningful is the content? (0.0 to 1.0)
2. Production Quality: Technical production quality? (0.0 to 1.0)
3. Content Enjoyment: How enjoyable is the content? (0.0 to 1.0)
4. Production Complexity: Complexity of production? (0.0 to 1.0)

Finally, give an overall recommendation score from 0.0 to 1.0 on whether this track should be recommended to other listeners.

IMPORTANT: Format your response as JSON with these exact keys:
{
  "genre": "string",
  "tempo": "string",
  "key": "string",
  "instruments": "string",
  "production_style": "string",
  "mood": "string",
  "content_usefulness": 0.0,
  "production_quality": 0.0,
  "content_enjoyment": 0.0,
  "production_complexity": 0.0,
  "overall_recommendation_score": 0.0
}"""
```

### Why This Prompt Works

1. **Structured Output**: The JSON format requirement helps ensure parseable responses
2. **Multi-dimensional Scoring**: Gets detailed quality assessment, not just a single score
3. **Suno-Specific Context**: Mentions "Suno AI generated track" to give context
4. **Clear Instructions**: Explicit scoring ranges and format requirements

## Integration with Hybrid Recommender

### Option 1: Use Precomputed Scores

```python
from sunorecsys.recommenders.hybrid import HybridRecommender

# Initialize with Music Flamingo quality scores
recommender = HybridRecommender(
    quality_weight=0.10,  # Weight for quality channel
    quality_threshold=0.4,  # Minimum quality score
    use_quality_filter=True,
)

# Fit with songs and quality scores file
recommender.fit(
    songs_df,
    quality_scores_file="data/music_flamingo_scores.json",
)
```

### Option 2: Use Music Flamingo Directly (Slower)

```python
from sunorecsys.recommenders.quality_filter import QualityFilter
from sunorecsys.utils.music_flamingo_quality import MusicFlamingoQualityScorer

# Initialize Music Flamingo scorer
scorer = MusicFlamingoQualityScorer(
    model_id="nvidia/music-flamingo-hf",
    device="cuda",  # or "cpu"
)

# Create quality filter with Music Flamingo
quality_filter = QualityFilter(
    quality_threshold=0.4,
    use_music_flamingo=True,
    music_flamingo_model_id="nvidia/music-flamingo-hf",
    device="cuda",
)

# Fit (will compute scores on-the-fly)
quality_filter.fit(songs_df)
```

## Caching Strategy

Quality scores are cached to avoid re-computation:

```
data/music_flamingo_scores/
├── song-1.json
├── song-2.json
└── song-3.json
```

Each cache file contains:
```json
{
  "content_usefulness": 0.85,
  "production_quality": 0.92,
  "content_enjoyment": 0.78,
  "production_complexity": 0.65,
  "overall_recommendation_score": 0.82
}
```

## Performance Considerations

### Processing Time
- **Per Song**: ~5-30 seconds (depends on audio length, model loading, GPU/CPU)
- **With GPU (CUDA)**: ~5-10 seconds per song
- **With CPU**: ~20-30 seconds per song
- **Batch Processing**: Music Flamingo processes one song at a time

### Recommendations
1. **Pre-compute Scores**: Run the script once to generate all scores
2. **Use Caching**: Always enable caching to avoid re-processing
3. **Use GPU**: Significantly faster on GPU (NVIDIA A100/H100 recommended)
4. **Process in Batches**: Can run overnight for large datasets

## File Format

The output JSON file has this structure:

```json
{
  "song-id-1": {
    "song_id": "song-id-1",
    "title": "Song Title",
    "scores": {
      "content_usefulness": 0.85,
      "production_quality": 0.92,
      "content_enjoyment": 0.78,
      "production_complexity": 0.65,
      "overall_recommendation_score": 0.82,
      "overall_quality_score": 0.82
    }
  },
  "song-id-2": {
    ...
  }
}
```

## Troubleshooting

### Issue: Model download fails
**Solution**: 
- Check internet connection
- Verify HuggingFace access
- Model size: ~8B parameters (~16GB)

### Issue: Out of memory
**Solution**: 
- Use CPU instead of GPU: `--device cpu`
- Process fewer songs: `--limit 10`
- Reduce max tokens: `--max-tokens 256`

### Issue: Slow processing
**Solution**: 
- Use GPU: `--device cuda`
- Process during off-hours
- Use precomputed scores instead of on-the-fly scoring

### Issue: JSON parsing fails
**Solution**: 
- The parser has fallback mechanisms
- Check individual cache files if scores seem wrong
- Model responses may vary - parser handles multiple formats

## Example: Complete Pipeline

```python
# Step 1: Compute quality scores (run once)
# python compute_music_flamingo_quality.py \
#     --input sunorecsys/data/curl/all_songs.json \
#     --output data/music_flamingo_scores.json

# Step 2: Use in hybrid recommender
from sunorecsys.recommenders.hybrid import HybridRecommender
import pandas as pd

# Load processed songs
songs_df = pd.read_parquet("data/processed/songs.parquet")

# Initialize recommender with Music Flamingo quality scores
recommender = HybridRecommender(
    item_cf_weight=0.25,
    item_content_weight=0.30,
    prompt_weight=0.20,
    user_weight=0.20,
    quality_weight=0.10,  # Quality filter weight
    quality_threshold=0.4,  # Minimum quality
    use_quality_filter=True,
)

# Fit with quality scores
recommender.fit(
    songs_df,
    quality_scores_file="data/music_flamingo_scores.json",
)

# Get recommendations (quality filter automatically applied)
recommendations = recommender.recommend(
    song_ids=["seed-song-1", "seed-song-2"],
    n=10
)

# Recommendations are already filtered by quality
print(f"Got {len(recommendations)} high-quality recommendations")
```

## Combining with Meta Audiobox Aesthetics

You can combine Music Flamingo with Meta Audiobox Aesthetics for even better quality filtering:

```python
# Compute both scores and combine
music_flamingo_scores = load_scores("data/music_flamingo_scores.json")
audiobox_scores = load_scores("data/audiobox_scores.json")

# Combine (example: 60% Music Flamingo, 40% Audiobox)
combined_scores = {
    song_id: 0.6 * mf_score + 0.4 * ab_score
    for song_id, (mf_score, ab_score) in zip(
        music_flamingo_scores.keys(),
        zip(music_flamingo_scores.values(), audiobox_scores.values())
    )
}
```

## Next Steps

1. **Tune Threshold**: Adjust `quality_threshold` based on your dataset
2. **Adjust Weights**: Modify weight distribution in `get_overall_quality_score()`
3. **Fine-tune Prompt**: Customize the prompt for your specific needs
4. **Batch Processing**: Process large datasets in batches
5. **Evaluate Impact**: A/B test with/without Music Flamingo filtering

## References

- [Music Flamingo GitHub](https://github.com/NVIDIA/audio-flamingo/tree/music_flamingo)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Music Flamingo Paper](https://research.nvidia.com/labs/adlr/AF3/)

