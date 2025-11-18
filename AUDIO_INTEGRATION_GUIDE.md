# Audio Feature Integration Guide

This guide explains how to integrate audio feature extraction into the recommendation system.

## Overview

The system now supports extracting audio features from audio URLs in the song data. This provides a direct signal of musical similarity that complements text-based approaches.

## Features Extracted

### Low-Level Features (librosa)

1. **MFCC (Mel-Frequency Cepstral Coefficients)**
   - 13 coefficients (mean + std = 26 dimensions)
   - Captures timbre and texture
   - Most important for musical similarity

2. **Chroma**
   - 12 pitch classes (mean + std = 24 dimensions)
   - Captures harmonic content
   - Useful for chord progressions

3. **Spectral Features**
   - Spectral centroid (brightness)
   - Spectral rolloff (frequency distribution)
   - Spectral bandwidth
   - Zero crossing rate
   - Total: 4 dimensions

4. **Rhythm Features**
   - Tempo (BPM)
   - Beat frames
   - Total: 2 dimensions

5. **Energy/Dynamics**
   - RMS (Root Mean Square) mean and std
   - Total: 2 dimensions

6. **Tonnetz**
   - Harmonic relationships
   - 6 dimensions

**Total Feature Dimension: 64**

## Usage

### Basic Usage

```python
from sunorecsys.data.preprocess import SongDataProcessor

# Enable audio feature extraction
processor = SongDataProcessor(extract_audio_features=True)

# Process songs (will download and extract audio features)
songs = processor.load_songs("all_songs.json")
songs_df = processor.process_all(songs, verbose=True)

# Audio features are stored in 'audio_features' column
print(songs_df[['song_id', 'has_audio_features']].head())
```

### Standalone Audio Extraction

```python
from sunorecsys.utils.audio_features import AudioFeatureExtractor

extractor = AudioFeatureExtractor(cache_dir="data/audio_cache")

# Extract features from URL
features = extractor.extract_features_from_url(
    audio_url="https://cdn1.suno.ai/song.mp3",
    song_id="song-123"
)

# Convert to vector
feature_vector = extractor.features_to_vector(features)
print(f"Feature dimension: {len(feature_vector)}")  # 64
```

### Batch Processing

```python
# Extract features for multiple songs
audio_urls = songs_df['audio_url'].tolist()
song_ids = songs_df['song_id'].tolist()

features_dict = extractor.extract_features_batch(
    audio_urls,
    song_ids,
    show_progress=True
)
```

## Integration with Recommenders

### Option 1: Extend Item-Based Recommender

```python
from sunorecsys.recommenders.item_based import ItemBasedRecommender
import numpy as np

# Load songs with audio features
songs_df = pd.read_parquet("data/processed/songs.parquet")

# Filter songs with audio features
songs_with_audio = songs_df[songs_df['has_audio_features'] == True].copy()

# Extract audio feature vectors
audio_vectors = np.array([
    np.array(features) for features in songs_with_audio['audio_features']
])

# Normalize audio features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
audio_vectors_normalized = scaler.fit_transform(audio_vectors)

# Train recommender with audio features
recommender = ItemBasedRecommender(
    tag_weight=0.3,
    prompt_weight=0.3,
    metadata_weight=0.1,
    audio_weight=0.3,  # New parameter
)

# Modify fit() to include audio features
recommender.fit(songs_with_audio, audio_features=audio_vectors_normalized)
```

### Option 2: Create Audio-Based Recommender

```python
from sunorecsys.recommenders.base import BaseRecommender
from sklearn.metrics.pairwise import cosine_similarity

class AudioBasedRecommender(BaseRecommender):
    """Recommender based solely on audio features"""
    
    def __init__(self):
        super().__init__("AudioBased")
        self.audio_features = None
        self.songs_df = None
    
    def fit(self, songs_df, audio_features):
        self.songs_df = songs_df
        self.audio_features = audio_features
        self.is_fitted = True
    
    def recommend(self, song_ids, n=10):
        # Get audio features of seed songs
        seed_indices = [self.songs_df.index[self.songs_df['song_id'] == sid][0] 
                       for sid in song_ids]
        seed_features = self.audio_features[seed_indices]
        avg_features = seed_features.mean(axis=0)
        
        # Compute cosine similarity
        similarities = cosine_similarity(
            avg_features.reshape(1, -1),
            self.audio_features
        )[0]
        
        # Get top N
        top_indices = np.argsort(similarities)[::-1][:n]
        return self._format_recommendations(...)
```

## Caching Strategy

Audio files and features are cached to avoid re-downloading and re-processing:

```
data/audio_cache/
├── audio/              # Downloaded audio files
│   ├── song-123_abc.mp3
│   └── song-456_def.mp3
└── features/          # Extracted features
    ├── song-123.pkl
    └── song-456.pkl
```

## Performance Considerations

### Download Time
- Average song (3-4 minutes): ~2-5 MB
- Download time: ~1-3 seconds per song (depends on connection)
- For 1000 songs: ~15-50 minutes total

### Processing Time
- Feature extraction: ~0.5-1 second per song (librosa)
- For 1000 songs: ~8-17 minutes total

### Storage
- Audio files: ~2-5 MB per song
- Features: ~1-2 KB per song (compressed)
- For 1000 songs: ~2-5 GB audio, ~1-2 MB features

### Recommendations
1. **Start Small**: Process a subset first (e.g., 100-500 songs)
2. **Use Caching**: Always enable caching to avoid re-processing
3. **Parallel Processing**: Can parallelize downloads and feature extraction
4. **Incremental**: Process new songs incrementally

## Advanced: Deep Audio Embeddings

For richer representations, consider using pre-trained models:

### CLAP (Contrastive Language-Audio Pretraining)
```python
# Future implementation
from transformers import CLAPProcessor, CLAPModel

processor = CLAPProcessor.from_pretrained("laion/larger_clap_music_and_speech")
model = CLAPModel.from_pretrained("laion/larger_clap_music_and_speech")

# Get audio embedding
audio_input = processor(audios=audio_path, return_tensors="pt")
audio_embedding = model.get_audio_features(**audio_input)
```

### MusicLM Embeddings
```python
# Google's MusicLM (when available)
# Provides music-specific embeddings
```

## Integration with Hybrid Recommender

To add audio features to the hybrid recommender:

1. **Extract audio features during preprocessing**
2. **Add AudioBasedRecommender as a new channel**
3. **Update hybrid weights**:
   ```python
   recommender = HybridRecommender(
       item_weight=0.20,
       prompt_weight=0.15,
       genre_weight=0.15,
       user_weight=0.20,
       audio_weight=0.20,  # New channel
       quality_weight=0.10,
   )
   ```

## Example: Complete Pipeline

```python
# Step 1: Preprocess with audio features
from sunorecsys.data.preprocess import SongDataProcessor

processor = SongDataProcessor(extract_audio_features=True)
songs = processor.load_songs("all_songs.json")
songs_df = processor.process_all(songs, verbose=True)
processor.save_processed("data/processed")

# Step 2: Train recommender with audio
from sunorecsys.recommenders.hybrid import HybridRecommender

# Load processed data
songs_df = processor.load_processed("data/processed")

# Filter songs with audio features
songs_with_audio = songs_df[songs_df['has_audio_features'] == True]

# Train (audio features are already in the dataframe)
recommender = HybridRecommender()
recommender.fit(songs_with_audio)

# Step 3: Use for recommendations
recommendations = recommender.recommend(
    song_ids=["seed-song-1", "seed-song-2"],
    n=10
)
```

## Troubleshooting

### Issue: Audio download fails
- **Solution**: Check URL validity, network connection, rate limits
- **Workaround**: Skip failed downloads, retry later

### Issue: Feature extraction slow
- **Solution**: Use caching, process in batches, consider parallelization
- **Workaround**: Start with subset of songs

### Issue: Memory issues
- **Solution**: Process in batches, clear cache periodically
- **Workaround**: Use streaming audio processing

### Issue: Incompatible audio formats
- **Solution**: librosa handles most formats, but may need conversion
- **Workaround**: Pre-convert audio to standard format (MP3, WAV)

## Next Steps

1. **Implement deep audio embeddings** (CLAP, MusicLM)
2. **Add temporal features** (song structure, evolution)
3. **Integrate with quality filter** (audio quality assessment)
4. **A/B test audio vs. text features** (measure improvement)

