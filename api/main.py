"""FastAPI service for music recommender system"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import joblib
from pathlib import Path

from sunorecsys.recommenders.hybrid import HybridRecommender

app = FastAPI(
    title="Suno Music Recommender API",
    description="Production-ready music recommendation service",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance
recommender: Optional[HybridRecommender] = None


class RecommendationRequest(BaseModel):
    user_id: Optional[str] = None
    song_ids: Optional[List[str]] = None
    n: int = 10
    exclude_song_ids: Optional[List[str]] = None


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    count: int


@app.on_event("startup")
async def load_recommender():
    """Load recommender model on startup"""
    global recommender
    
    model_path = Path("model_checkpoints/hybrid_recommender.pkl")
    if model_path.exists():
        try:
            recommender = HybridRecommender.load(str(model_path))
            print(f"✅ Loaded recommender from {model_path}")
        except Exception as e:
            print(f"⚠️  Failed to load recommender: {e}")
    else:
        print(f"⚠️  Model not found at {model_path}. Please train the model first.")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Suno Music Recommender",
        "model_loaded": recommender is not None and recommender.is_fitted,
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if recommender and recommender.is_fitted else "unhealthy",
        "model_loaded": recommender is not None and recommender.is_fitted,
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Generate music recommendations
    
    Args:
        request: Recommendation request with user_id or song_ids
    
    Returns:
        List of recommended songs with scores
    """
    if not recommender or not recommender.is_fitted:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    
    if not request.user_id and not request.song_ids:
        raise HTTPException(
            status_code=400,
            detail="Either user_id or song_ids must be provided"
        )
    
    try:
        recommendations = recommender.recommend(
            user_id=request.user_id,
            song_ids=request.song_ids,
            n=request.n,
            exclude_song_ids=request.exclude_song_ids,
        )
        
        return RecommendationResponse(
            recommendations=recommendations,
            count=len(recommendations),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend", response_model=RecommendationResponse)
async def recommend_get(
    user_id: Optional[str] = Query(None, description="User ID"),
    song_ids: Optional[str] = Query(None, description="Comma-separated song IDs"),
    n: int = Query(10, ge=1, le=100, description="Number of recommendations"),
    exclude_song_ids: Optional[str] = Query(None, description="Comma-separated song IDs to exclude"),
):
    """GET endpoint for recommendations"""
    song_ids_list = song_ids.split(",") if song_ids else None
    exclude_list = exclude_song_ids.split(",") if exclude_song_ids else None
    
    request = RecommendationRequest(
        user_id=user_id,
        song_ids=song_ids_list,
        n=n,
        exclude_song_ids=exclude_list,
    )
    
    return await recommend(request)


@app.get("/similar/{song_id}")
async def get_similar_songs(
    song_id: str,
    n: int = Query(10, ge=1, le=100),
):
    """Get similar songs to a given song"""
    if not recommender or not recommender.is_fitted:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    
    try:
        similar = recommender.get_similar_songs(song_id, n=n)
        return {
            "song_id": song_id,
            "similar_songs": similar,
            "count": len(similar),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def get_info():
    """Get information about the recommender"""
    if not recommender or not recommender.is_fitted:
        return {
            "status": "not_loaded",
            "message": "Recommender not loaded",
        }
    
    return {
        "name": recommender.name,
        "item_weight": recommender.item_weight,
        "prompt_weight": recommender.prompt_weight,
        "genre_weight": recommender.genre_weight,
        "user_weight": recommender.user_weight,
        "quality_weight": recommender.quality_weight,
        "total_songs": len(recommender.songs_df) if recommender.songs_df is not None else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

