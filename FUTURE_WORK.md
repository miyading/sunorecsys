# Future Work

## Scalability Improvements

### ALS Matrix Factorization for User-Based CF

**Current Implementation**: User-user similarity matrix based CF
- Uses precomputed user-user similarity matrix
- Computes cosine similarity between all user pairs
- Memory: O(num_users²) for similarity matrix

**Future Work**: ALS Matrix Factorization
- **Why**: Better scalability for large user bases
- **Memory**: O(num_users × factors + num_items × factors) instead of O(num_users²)
- **Benefits**:
  - Handles millions of users efficiently
  - Faster training for sparse data
  - Can incorporate implicit feedback weights (play counts, etc.)
  
**Implementation Plan**:
```python
# Future: Use implicit library for ALS
from implicit.als import AlternatingLeastSquares

self.model = AlternatingLeastSquares(
    factors=50,  # Latent dimension
    regularization=0.1,
    iterations=15,
)
self.model.fit(self.user_item_matrix.T)  # Implicit expects item-user matrix

# Recommendation: user_factors @ item_factors.T
# Much more memory efficient for large user bases
```

**When to Switch**:
- When user base > 100K users
- When similarity matrix memory becomes a bottleneck
- When training time becomes too slow

**Trade-offs**:
- ✅ Better scalability
- ✅ Lower memory footprint
- ❌ Less interpretable (latent factors vs explicit similarity)
- ❌ Slightly slower inference for small user bases

## Other Future Enhancements

### 1. Weighted User-User Similarity
- Currently: Binary interactions (0/1)
- Future: Weight by play counts, recency, etc.
- Formula: `similarity = weighted_cosine(user_vector_1, user_vector_2)`

### 2. Incremental Similarity Updates
- Currently: Recompute entire similarity matrix weekly
- Future: Incremental updates for new users/interactions
- Benefit: Faster weekly updates

### 3. Approximate Similarity Search
- Currently: Exact cosine similarity for all users
- Future: Use approximate nearest neighbors (e.g., Annoy, FAISS)
- Benefit: O(log n) query time vs O(n) for large user bases

### 4. Distributed Computing
- Future: Parallel similarity computation
- Use Spark or Dask for large-scale processing
- Benefit: Handle billions of users

## UI/UX Design

### Interactive Recommendation Visualization

**Goal**: Create an engaging, visual interface that helps users understand why they're seeing specific recommendations.

#### 1. User-Based CF Visualization: Floating User Bubbles

**Concept**: Show similar users as floating, animated bubbles around the current user's profile.

**Design Elements**:
- **Central User Avatar**: The current user's profile picture/icon in the center
- **Floating Bubbles**: Similar users' avatars floating around the center user
  - Bubble size: Proportional to similarity score (more similar = larger bubble)
  - Bubble position: Animated, bouncing around the center user
  - Bubble color: Gradient based on similarity (red = high similarity, blue = lower similarity)
  - Hover effect: Show user name, similarity score, and top shared songs
- **Animation**:
  - Gentle floating/bouncing motion
  - Bubbles slowly orbit around the center user
  - Smooth transitions when recommendations update
- **Interaction**:
  - Click on a bubble to see that user's listening history
  - Show connection lines between center user and bubbles (thickness = similarity)
  - Highlight bubbles that contributed to current recommendations

**Technical Implementation**:
```javascript
// Pseudo-code for bubble animation
class UserBubble {
  constructor(userId, similarity, position) {
    this.userId = userId;
    this.similarity = similarity;
    this.radius = similarity * 50; // Scale bubble size
    this.position = position;
    this.velocity = randomVelocity();
  }
  
  update() {
    // Bouncing physics
    this.position += this.velocity;
    this.velocity += gravity;
    // Boundary collision detection
    // Smooth animation with requestAnimationFrame
  }
  
  render() {
    // Draw bubble with user avatar
    // Apply color gradient based on similarity
    // Show connection line to center user
  }
}
```

**Visual Style**:
- Modern, clean design with glassmorphism effects
- Smooth animations (60fps)
- Responsive to screen size
- Dark/light theme support

#### 2. Item-Based CF Visualization: Rotating Record Player

**Concept**: Show recommended tracks as records rotating on a turntable, with the user's listening history as the "spindle" (center).

**Design Elements**:
- **Center Spindle**: User's last-n interacted songs displayed as stacked records
  - Each record represents a seed song from user history
  - Records are stacked vertically in the center
  - Click to see song details
- **Rotating Records**: Recommended songs as records rotating around the spindle
  - Each record shows album art or song cover
  - Rotation speed: Based on recommendation score (higher score = faster rotation)
  - Position: Arranged in a circle around the center
  - Size: Proportional to recommendation score
- **Connection Lines**: Visual lines connecting seed songs (center) to recommendations
  - Line thickness: Based on similarity score
  - Line color: Gradient from seed song to recommendation
  - Animated flow effect showing the recommendation path
- **Animation**:
  - Records continuously rotate (like a turntable)
  - Smooth, realistic rotation physics
  - Gentle wobble effect for added realism
  - Records can be "picked up" and examined (hover/click)

**Technical Implementation**:
```javascript
// Pseudo-code for record player visualization
class RecordPlayer {
  constructor(seedSongs, recommendations) {
    this.seedSongs = seedSongs; // Center spindle
    this.recommendations = recommendations; // Rotating records
  }
  
  render() {
    // Draw center spindle with stacked seed songs
    this.seedSongs.forEach((song, index) => {
      drawRecord(song, centerX, centerY, index * recordThickness);
    });
    
    // Draw rotating recommendation records
    this.recommendations.forEach((rec, index) => {
      const angle = (index / this.recommendations.length) * 2 * Math.PI;
      const radius = baseRadius + rec.score * radiusMultiplier;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      
      drawRotatingRecord(rec, x, y, rec.rotationSpeed);
      drawConnectionLine(seedSong, rec);
    });
  }
}
```

**Visual Style**:
- Retro-inspired design with modern touches
- Realistic record textures and shadows
- Smooth rotation animations
- Interactive hover states showing song details
- Responsive layout that adapts to number of recommendations

#### 3. Combined Visualization

**Concept**: Integrate both visualizations in a unified interface.

**Layout**:
- **Left Panel**: User-based CF visualization (floating bubbles)
- **Right Panel**: Item-based CF visualization (rotating records)
- **Center**: Current user profile and recommendation summary
- **Bottom**: Detailed recommendation list with source attribution

**Features**:
- Synchronized interactions (selecting a bubble highlights related records)
- Cross-visualization connections
- Real-time updates as user interacts with recommendations
- Mobile-responsive design with simplified animations

**Future Enhancements**:
- 3D visualization option
- VR/AR support for immersive experience
- Social features (share recommendations, see friends' visualizations)
- Customizable themes and animation speeds

