"""Evaluation metrics for recommender systems"""

import numpy as np
from typing import List, Dict, Set, Any
from collections import defaultdict


def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Precision@K"""
    if k == 0:
        return 0.0
    
    top_k = recommended[:k]
    relevant_in_top_k = len([item for item in top_k if item in relevant])
    return relevant_in_top_k / k


def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Recall@K"""
    if len(relevant) == 0:
        return 0.0
    
    top_k = recommended[:k]
    relevant_in_top_k = len([item for item in top_k if item in relevant])
    return relevant_in_top_k / len(relevant)


def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain@K"""
    if len(relevant) == 0:
        return 0.0
    
    top_k = recommended[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    num_relevant = min(len(relevant), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def map_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Mean Average Precision@K"""
    if len(relevant) == 0:
        return 0.0
    
    top_k = recommended[:k]
    relevant_positions = [i + 1 for i, item in enumerate(top_k) if item in relevant]
    
    if not relevant_positions:
        return 0.0
    
    # Calculate average precision
    precisions = []
    for pos in relevant_positions:
        precisions.append(len([p for p in relevant_positions if p <= pos]) / pos)
    
    return np.mean(precisions)


def diversity(recommended: List[str], song_features: Dict[str, Any], metric: str = 'genre') -> float:
    """
    Calculate diversity of recommendations
    
    Args:
        recommended: List of recommended song IDs
        song_features: Dict mapping song_id to features (e.g., genre, tags)
        metric: Feature to use for diversity ('genre' or 'tags')
    
    Returns:
        Diversity score (0-1, higher is more diverse)
    """
    if len(recommended) <= 1:
        return 1.0
    
    features = []
    for song_id in recommended:
        if song_id in song_features:
            if metric == 'genre':
                features.append(song_features[song_id].get('genre', 'unknown'))
            elif metric == 'tags':
                tags = song_features[song_id].get('tags', [])
                features.extend(tags if isinstance(tags, list) else [])
    
    if not features:
        return 0.0
    
    # Calculate unique features ratio
    unique_features = len(set(features))
    total_features = len(features)
    
    return unique_features / total_features if total_features > 0 else 0.0


def coverage(recommended: List[str], catalog: Set[str]) -> float:
    """Calculate catalog coverage"""
    if len(catalog) == 0:
        return 0.0
    
    unique_recommended = set(recommended)
    return len(unique_recommended) / len(catalog)


def evaluate_recommender(
    recommender,
    test_data: List[Dict[str, Any]],
    k_values: List[int] = [5, 10, 20],
    songs_df=None,
) -> Dict[str, Any]:
    """
    Evaluate a recommender on test data
    
    Args:
        recommender: Recommender instance with recommend() method
        test_data: List of test cases, each with 'user_id' or 'song_ids' and 'relevant' (set of relevant song IDs)
        k_values: List of k values for metrics
        songs_df: Optional DataFrame for diversity calculation
    
    Returns:
        Dictionary of evaluation metrics
    """
    results = {
        'precision': defaultdict(list),
        'recall': defaultdict(list),
        'ndcg': defaultdict(list),
        'map': defaultdict(list),
    }
    
    song_features = {}
    if songs_df is not None:
        for _, row in songs_df.iterrows():
            song_features[row['song_id']] = {
                'genre': row.get('genre'),
                'tags': row.get('tags', []),
            }
    
    all_recommended = []
    
    for test_case in test_data:
        # Get recommendations
        if 'user_id' in test_case:
            recommendations = recommender.recommend(
                user_id=test_case['user_id'],
                n=max(k_values) * 2,
            )
        elif 'song_ids' in test_case:
            recommendations = recommender.recommend(
                song_ids=test_case['song_ids'],
                n=max(k_values) * 2,
            )
        else:
            continue
        
        recommended_ids = [r['song_id'] for r in recommendations]
        relevant = test_case.get('relevant', set())
        
        if len(relevant) == 0:
            continue
        
        all_recommended.extend(recommended_ids)
        
        # Calculate metrics for each k
        for k in k_values:
            results['precision'][k].append(precision_at_k(recommended_ids, relevant, k))
            results['recall'][k].append(recall_at_k(recommended_ids, relevant, k))
            results['ndcg'][k].append(ndcg_at_k(recommended_ids, relevant, k))
            results['map'][k].append(map_at_k(recommended_ids, relevant, k))
    
    # Aggregate results
    aggregated = {}
    for metric_name, metric_results in results.items():
        aggregated[metric_name] = {}
        for k in k_values:
            if metric_results[k]:
                aggregated[metric_name][f'@{k}'] = {
                    'mean': np.mean(metric_results[k]),
                    'std': np.std(metric_results[k]),
                }
    
    # Calculate diversity and coverage
    if songs_df is not None:
        aggregated['diversity'] = {
            'genre': diversity(all_recommended, song_features, 'genre'),
            'tags': diversity(all_recommended, song_features, 'tags'),
        }
        
        catalog = set(songs_df['song_id'].unique())
        aggregated['coverage'] = coverage(all_recommended, catalog)
    
    return aggregated

