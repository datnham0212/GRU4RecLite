"""
Simple baseline recommendation models for comparison.
- MostPopularBaseline: Always recommend the K most popular items globally
- LastItemBaseline: Recommend items similar to the user's last clicked item
"""

import numpy as np
import pandas as pd
from collections import Counter


class MostPopularBaseline:
    """
    Recommend the K globally most popular items.
    Popularity is measured by occurrence frequency in the training data.
    """
    def __init__(self, item_key='item_id', session_key='session_id'):
        self.item_key = item_key
        self.session_key = session_key
        self.popular_items = []
        self.item_counts = None

    def fit(self, train_data):
        """
        Train on training data: count item frequencies.
        
        Args:
            train_data: pd.DataFrame with columns [session_key, item_key, ...]
        """
        # Count occurrences of each item
        self.item_counts = train_data[self.item_key].value_counts()
        self.popular_items = self.item_counts.index.tolist()
        return self

    def recommend(self, session_items, topk=20, exclude_seen=True):
        """
        Return top-K most popular items.
        
        Args:
            session_items: list of item_ids in current session (for exclude_seen)
            topk: number of recommendations
            exclude_seen: if True, exclude items already in session
            
        Returns:
            list of recommended item_ids (as strings)
        """
        if not self.popular_items:
            return []

        if exclude_seen and session_items:
            # Filter out seen items
            session_set = set(session_items)
            candidates = [item for item in self.popular_items if item not in session_set]
            return [str(item) for item in candidates[:topk]]
        else:
            return [str(item) for item in self.popular_items[:topk]]


class LastItemBaseline:
    """
    Recommend items similar to the user's last clicked item.
    Similarity is computed using embedding cosine distance.
    Falls back to MostPopular if embeddings unavailable.
    """
    def __init__(self, item_embeddings=None, item_key='item_id', session_key='session_id'):
        """
        Args:
            item_embeddings: dict or Series mapping item_id -> embedding vector (np.ndarray)
                            or None (will be set during fit)
            item_key: column name for item IDs
            session_key: column name for session IDs
        """
        self.item_key = item_key
        self.session_key = session_key
        self.item_embeddings = item_embeddings
        self.fallback = MostPopularBaseline(item_key, session_key)

    def fit(self, train_data, item_embeddings=None):
        """
        Train fallback and optionally set embeddings.
        
        Args:
            train_data: pd.DataFrame with columns [session_key, item_key, ...]
            item_embeddings: dict or Series mapping item_id -> embedding vector
        """
        # Fit fallback
        self.fallback.fit(train_data)
        
        # Set embeddings if provided
        if item_embeddings is not None:
            if isinstance(item_embeddings, dict):
                self.item_embeddings = item_embeddings
            elif isinstance(item_embeddings, pd.Series):
                self.item_embeddings = item_embeddings.to_dict()
        
        return self

    def recommend(self, session_items, topk=20, exclude_seen=True):
        """
        Recommend items similar to the last item in session.
        
        Args:
            session_items: list of item_ids in current session
            topk: number of recommendations
            exclude_seen: if True, exclude items already in session
            
        Returns:
            list of recommended item_ids (as strings)
        """
        # If no session items or no embeddings, fall back to MostPopular
        if not session_items or self.item_embeddings is None:
            return self.fallback.recommend(session_items, topk, exclude_seen)

        last_item = session_items[-1]
        
        # If last item has no embedding, fall back
        if last_item not in self.item_embeddings:
            return self.fallback.recommend(session_items, topk, exclude_seen)

        try:
            last_emb = self.item_embeddings[last_item]
            
            # Compute cosine similarity with all items
            similarities = {}
            for item, emb in self.item_embeddings.items():
                if isinstance(emb, np.ndarray) and isinstance(last_emb, np.ndarray):
                    # Cosine similarity
                    dot = np.dot(last_emb, emb)
                    norm_last = np.linalg.norm(last_emb)
                    norm_item = np.linalg.norm(emb)
                    if norm_last > 0 and norm_item > 0:
                        similarities[item] = dot / (norm_last * norm_item)
                    else:
                        similarities[item] = 0.0
            
            # Sort by similarity (descending)
            sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Filter and return
            if exclude_seen:
                session_set = set(session_items)
                candidates = [item for item, _ in sorted_items if item not in session_set]
            else:
                candidates = [item for item, _ in sorted_items]
            
            return [str(item) for item in candidates[:topk]]
        
        except Exception:
            # On any error, fall back to MostPopular
            return self.fallback.recommend(session_items, topk, exclude_seen)
