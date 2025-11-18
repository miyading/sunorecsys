"""Embedding utilities for text and audio"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import joblib
from pathlib import Path


class TextEmbedder:
    """Text embedding model for prompts and tags"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[str] = None):
        """
        Initialize text embedder
        
        Args:
            model_name: HuggingFace model name or path
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        non_empty = [t if t else " " for t in texts]
        
        embeddings = self.model.encode(
            non_empty,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )
        
        return embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not text:
            text = " "
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    
    def save(self, path: str):
        """Save the embedder"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
        }, path)
        # Save the actual model separately
        model_path = Path(path).parent / f"{Path(path).stem}_model"
        self.model.save(str(model_path))
    
    @classmethod
    def load(cls, path: str, cache_dir: Optional[str] = None):
        """Load the embedder"""
        data = joblib.load(path)
        embedder = cls(data['model_name'], cache_dir=cache_dir)
        return embedder


class TagEmbedder:
    """Specialized embedder for music tags"""
    
    def __init__(self, base_embedder: TextEmbedder):
        self.base_embedder = base_embedder
    
    def embed_tags(self, tags: List[str]) -> np.ndarray:
        """Embed a list of tags by joining them"""
        if not tags:
            return self.base_embedder.embed_text("")
        
        # Join tags with commas
        tag_str = ", ".join(tags)
        return self.base_embedder.embed_text(tag_str)
    
    def embed_tag_list(self, tag_lists: List[List[str]], show_progress: bool = False) -> np.ndarray:
        """Embed multiple tag lists"""
        tag_strings = [", ".join(tags) if tags else "" for tags in tag_lists]
        return self.base_embedder.embed_texts(tag_strings, show_progress=show_progress)


class PromptEmbedder:
    """Specialized embedder for generation prompts"""
    
    def __init__(self, base_embedder: TextEmbedder):
        self.base_embedder = base_embedder
    
    def embed_prompt(self, prompt: str) -> np.ndarray:
        """Embed a generation prompt"""
        # For long prompts, we might want to truncate or use a different strategy
        # For now, use the full prompt
        return self.base_embedder.embed_text(prompt)
    
    def embed_prompts(self, prompts: List[str], show_progress: bool = False) -> np.ndarray:
        """Embed multiple prompts"""
        return self.base_embedder.embed_texts(prompts, show_progress=show_progress)

