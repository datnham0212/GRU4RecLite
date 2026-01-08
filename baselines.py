"""
Simple baseline recommendation models for comparison.
- MostPopularBaseline: Always recommend the K most popular items globally
- LastItemBaseline: Recommend items similar to the user's last clicked item
- ItemKNNBaseline: Recommend items based on item-item similarity from co-occurrences
"""

import numpy as np
import pandas as pd
from collections import Counter
import time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


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


class ItemKNNBaseline:
    """
    Item-KNN: Recommend items based on similarity to items in current session.
    Similarity is computed using cosine distance between item co-occurrence vectors.
    """
    
    def __init__(self, k=20, min_support=1, item_key='item_id', session_key='session_id'):
        self.k = k
        self.min_support = min_support
        self.item_key = item_key
        self.session_key = session_key
        self.similarity_matrix = None
        self.item_to_idx = None
        self.idx_to_item = None
        self.n_items = 0
        
    def fit(self, data):
        """Fit ItemKNN by computing item-item similarity matrix."""
        print("  [ItemKNN] Computing item-item similarity...", end='', flush=True)
        start_time = time.time()
        
        # Create item mapping
        unique_items = sorted(data[self.item_key].unique())
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        self.n_items = len(self.item_to_idx)
        
        # Build co-occurrence matrix: rows=items, cols=sessions
        sessions = data.groupby(self.session_key)[self.item_key].apply(list)
        
        row_indices = []
        col_indices = []
        for session_idx, items in enumerate(sessions):
            for item in items:
                if item in self.item_to_idx:
                    row_indices.append(self.item_to_idx[item])
                    col_indices.append(session_idx)
        
        # Create sparse matrix (items × sessions)
        cooccurrence = csr_matrix(
            (np.ones(len(row_indices)), (row_indices, col_indices)),
            shape=(self.n_items, len(sessions))
        )
        
        # Normalize and compute cosine similarity
        cooccurrence_normalized = normalize(cooccurrence, norm='l2', axis=1)
        self.similarity_matrix = cosine_similarity(cooccurrence_normalized, dense_output=False)
        
        fit_time = time.time() - start_time
        print(f" done ({fit_time:.2f}s)")
        
    def recommend(self, session_items, topk=20, exclude_seen=True):
        """Recommend topk items for a session."""
        if not session_items:
            return []
        
        # Convert session items to indices
        session_indices = []
        for item in session_items:
            if item in self.item_to_idx:
                session_indices.append(self.item_to_idx[item])
        
        if not session_indices:
            return []
        
        # Compute recommendation scores: average similarity across session items
        scores = np.zeros(self.n_items)
        for session_idx in session_indices:
            if session_idx < self.n_items:
                scores += self.similarity_matrix[session_idx].toarray().ravel()
        scores /= len(session_indices)
        
        # Optionally mask out seen items
        if exclude_seen:
            for idx in session_indices:
                if idx < self.n_items:
                    scores[idx] = -np.inf
    
        # Get top-k
        top_indices = np.argsort(-scores)[:topk]
        
        # Convert back to item IDs
        recommendations = [str(self.idx_to_item[idx]) for idx in top_indices if idx < self.n_items]
        
        return recommendations
