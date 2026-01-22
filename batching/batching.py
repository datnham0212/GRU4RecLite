#!/usr/bin/env python
"""
GRU4REC SESSION-PARALLEL BATCHING
=================================

Implementation of GRU4Rec-style session-parallel batching.

Key properties:
- ONE item per session per step (not prefixes)
- Hidden state PERSISTS across batches (reset only when session ends)
- No padding, no attention masks
- Session replacement when finished
- Negative sampling support
- Weight tying (constrained embedding)

Usage:
    python batching.py [--epochs 10] [--batch-size 48] [--device cuda:0]
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(path, item_key='item_id', session_key='session_id', time_key='timestamp'):
    """
    Load session data from tab-separated file.
    Returns DataFrame sorted by session and time.
    """
    print(f"[DATA] Loading from {path}")
    data = pd.read_csv(path, sep='\t', dtype={session_key: 'int32', item_key: 'str'})
    
    # Sort by session, then by time
    data.sort_values([session_key, time_key], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    print(f"[DATA] Loaded {len(data)} events, {data[session_key].nunique()} sessions, {data[item_key].nunique()} items")
    return data


def create_item_map(data, item_key='item_id'):
    """Create item_id -> index mapping."""
    unique_items = data[item_key].unique()
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    return item_to_idx


# =============================================================================
# MODEL: GRU with Negative Sampling and Weight Tying
# =============================================================================

class SessionGRU(nn.Module):
    """
    GRU model for session-based recommendation.
    Uses GRUCell for step-by-step processing with persistent hidden states.
    Supports negative sampling and weight tying (constrained embedding).
    """
    
    def __init__(self, n_items, hidden_size=100, embedding_dim=0, n_layers=1, 
                 dropout_embed=0.0, dropout_hidden=0.0, constrained_embedding=True):
        super().__init__()
        
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim if embedding_dim > 0 else hidden_size
        self.constrained_embedding = constrained_embedding
        
        # Embedding layer
        self.embedding = nn.Embedding(n_items, self.embedding_dim, padding_idx=0)
        
        # GRU layers (using GRUCell for step-by-step control)
        self.gru_cells = nn.ModuleList()
        for i in range(n_layers):
            input_size = self.embedding_dim if i == 0 else hidden_size
            self.gru_cells.append(nn.GRUCell(input_size, hidden_size))
        
        # Output layer (weight-tied or separate)
        if constrained_embedding:
            # Weight tying: output uses embedding weights transposed
            self.output_bias = nn.Parameter(torch.zeros(n_items))
        else:
            self.output = nn.Linear(hidden_size, n_items)
        
        # Dropout
        self.dropout_embed = nn.Dropout(dropout_embed)
        self.dropout_hidden = nn.Dropout(dropout_hidden)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_idx, hidden):
        """
        Process ONE item per session (true session-parallel step).
        
        Args:
            input_idx: (B,) tensor of item indices
            hidden: list of (B, hidden_size) tensors, one per layer
        
        Returns:
            logits: (B, n_items) output scores
            hidden: list of updated hidden states
        """
        # Embedding
        x = self.embedding(input_idx)  # (B, embedding_dim)
        x = self.dropout_embed(x)
        
        # Process through GRU layers
        new_hidden = []
        for i, gru_cell in enumerate(self.gru_cells):
            h = gru_cell(x, hidden[i])  # (B, hidden_size)
            new_hidden.append(h)
            x = self.dropout_hidden(h)
        
        # Output scores
        if self.constrained_embedding:
            # Weight tying: logits = h @ E^T + bias
            logits = torch.matmul(x, self.embedding.weight.t()) + self.output_bias
        else:
            logits = self.output(x)  # (B, n_items)
        
        return logits, new_hidden
    
    def forward_with_negatives(self, input_idx, hidden, target_idx, negative_idx):
        """
        Forward pass with negative sampling.
        Only computes scores for target + negative samples.
        
        Args:
            input_idx: (B,) tensor of item indices
            hidden: list of (B, hidden_size) tensors
            target_idx: (B,) tensor of target item indices
            negative_idx: (B, n_neg) tensor of negative sample indices
        
        Returns:
            target_scores: (B,) scores for targets
            negative_scores: (B, n_neg) scores for negatives
            hidden: updated hidden states
        """
        # Embedding
        x = self.embedding(input_idx)  # (B, embedding_dim)
        x = self.dropout_embed(x)
        
        # Process through GRU layers
        new_hidden = []
        for i, gru_cell in enumerate(self.gru_cells):
            h = gru_cell(x, hidden[i])  # (B, hidden_size)
            new_hidden.append(h)
            x = self.dropout_hidden(h)
        
        # Get embeddings for targets and negatives
        target_emb = self.embedding(target_idx)  # (B, emb_dim)
        negative_emb = self.embedding(negative_idx)  # (B, n_neg, emb_dim)
        
        # Compute scores via dot product
        target_scores = (x * target_emb).sum(dim=1)  # (B,)
        negative_scores = torch.bmm(negative_emb, x.unsqueeze(2)).squeeze(2)  # (B, n_neg)
        
        # Add biases if weight tying
        if self.constrained_embedding:
            target_scores = target_scores + self.output_bias[target_idx]
            negative_scores = negative_scores + self.output_bias[negative_idx]
        
        return target_scores, negative_scores, new_hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden states for a batch."""
        return [torch.zeros(batch_size, self.hidden_size, device=device) 
                for _ in range(self.n_layers)]
    
    def step(self, input_idx, hidden):
        """
        Forward one step and return hidden representation only (no output layer).
        Used for TOP1 loss with in-batch negatives.
        
        Args:
            input_idx: (B,) tensor of item indices
            hidden: list of (B, hidden_size) tensors
        
        Returns:
            h: (B, hidden_size) final hidden representation
            new_hidden: list of updated hidden states
        """
        x = self.embedding(input_idx)
        x = self.dropout_embed(x)
        
        new_hidden = []
        for i, gru_cell in enumerate(self.gru_cells):
            h = gru_cell(x, hidden[i])
            new_hidden.append(h)
            x = self.dropout_hidden(h)
        
        return x, new_hidden


# =============================================================================
# NEGATIVE SAMPLER (FIXED: avoid padding_idx=0)
# =============================================================================

class NegativeSampler:
    """
    Popularity-based negative sampler.
    Samples items proportional to frequency^alpha.
    Supports logq correction for sampled softmax.
    """
    
    def __init__(self, item_counts, n_items, alpha=0.75, device='cpu'):
        self.n_items = n_items
        self.device = device
        self.alpha = alpha
        
        # Compute sampling probabilities: freq^alpha
        # Start from 1 to avoid padding_idx=0
        counts = np.zeros(n_items, dtype=np.float32)
        for item_idx, count in item_counts.items():
            if 0 < item_idx < n_items:  # Skip padding idx 0
                counts[item_idx] = count
        
        # Avoid zero counts (but keep idx 0 at zero prob)
        counts[1:] = counts[1:] + 1
        probs = np.power(counts, alpha)
        probs[0] = 0  # Never sample padding
        probs = probs / probs.sum()
        
        self.probs = probs
        self.items = np.arange(n_items)
        
        # Store log probabilities for logq correction
        # Use small epsilon to avoid log(0)
        self.log_probs = np.log(probs + 1e-10).astype(np.float32)
        self.log_probs_tensor = torch.tensor(self.log_probs, device=device)
    
    def sample(self, batch_size, n_samples, return_logq=False):
        """Sample negative items, optionally returning log probabilities."""
        neg_idx = np.random.choice(self.items, size=(batch_size, n_samples), p=self.probs)
        neg_idx_tensor = torch.tensor(neg_idx, dtype=torch.long, device=self.device)
        
        if return_logq:
            # Look up log probabilities for sampled items
            neg_logq = self.log_probs_tensor[neg_idx_tensor]  # (B, n_samples)
            return neg_idx_tensor, neg_logq
        return neg_idx_tensor
    
    def get_logq(self, item_idx):
        """Get log probability for given item indices (for targets)."""
        return self.log_probs_tensor[item_idx]


# =============================================================================
# SAMPLED SOFTMAX LOSS
# =============================================================================

def sampled_softmax_loss(target_scores, negative_scores, target_logq=None, negative_logq=None):
    """
    Compute sampled softmax loss.
    
    Args:
        target_scores: (B,) scores for positive targets
        negative_scores: (B, n_neg) scores for negative samples
    
    Returns:
        loss: scalar
    """
    # Apply logq correction if provided
    if target_logq is not None and negative_logq is not None:
        target_scores = target_scores - target_logq
        negative_scores = negative_scores - negative_logq
    
    # Concatenate: [target, negatives]
    # target at position 0, negatives at positions 1..n_neg
    all_scores = torch.cat([target_scores.unsqueeze(1), negative_scores], dim=1)  # (B, 1+n_neg)
    
    # Softmax cross-entropy: target is always at index 0
    labels = torch.zeros(all_scores.shape[0], dtype=torch.long, device=all_scores.device)
    loss = nn.functional.cross_entropy(all_scores, labels)
    
    return loss


def top1_loss(pos_scores, neg_scores):
    """
    TOP1 loss from the original GRU4Rec paper.
    
    Args:
        pos_scores: (B,) scores for positive targets
        neg_scores: (B, B-1) scores for in-batch negatives
    
    Returns:
        scalar loss
    """
    # sigmoid(s_j - s_i)
    diff = neg_scores - pos_scores.unsqueeze(1)
    term1 = torch.sigmoid(diff)
    
    # sigmoid(s_j)^2 (NOT sigmoid(s_j^2))
    term2 = torch.sigmoid(neg_scores) ** 2
    
    loss = term1 + term2
    return loss.mean()


# =============================================================================
# SESSION-PARALLEL DATA ITERATOR (TRUE GRU4REC STYLE)
# =============================================================================

class SessionParallelIterator:
    """
    TRUE GRU4Rec-style session-parallel iterator for training & evaluation.
    """
    def __init__(self, data, batch_size, item_to_idx,
                 item_key='item_id', session_key='session_id', time_key='timestamp',
                 device='cpu'):
        self.batch_size = batch_size
        self.device = device
        self.item_to_idx = item_to_idx
        self.item_key = item_key
        self.session_key = session_key
        
        data = data.copy()
        data.sort_values([session_key, time_key], inplace=True)
        
        self.session_ids = data[session_key].values
        self.item_ids = data[item_key].values
        self.items = np.array([item_to_idx.get(item, 0) for item in self.item_ids])
        
        session_change = np.concatenate([[True], self.session_ids[1:] != self.session_ids[:-1]])
        self.session_starts = np.where(session_change)[0]
        self.session_ends = np.concatenate([self.session_starts[1:], [len(self.items)]])
        self.n_sessions = len(self.session_starts)
        
        print(f"[ITERATOR] {self.n_sessions} sessions, {len(self.items)} events")

    def __call__(self, model, optimizer=None, training=True, neg_sampler=None, n_neg=2048, accum_steps=1, logq=0.0):
        """
        Yields one step per active session, maintaining persistent hidden states.

        Args:
            logq: logq correction weight (0.0 = no correction, 1.0 = full correction)

        Yields:
            input_idx: (n_active,) current items for active slots
            target_idx: (n_active,) next items for active slots
            logits: (n_active, n_items) output scores (None if training with neg sampling)
            loss_val: scalar loss (0.0 if not training)
            active_slots: (n_active,) indices of active slots in this step
        """
        batch_size = min(self.batch_size, self.n_sessions)
        slot_session = np.arange(batch_size)
        slot_pos = np.zeros(batch_size, dtype=np.int32)
        next_session_idx = batch_size
        
        hidden = model.init_hidden(batch_size, self.device)
        step_count = 0  # Track steps for gradient accumulation
        
        if training and optimizer is not None:
            optimizer.zero_grad()
        
        while True:
            # Check if ALL slots are exhausted (not ANY)
            if not (slot_session >= 0).any():
                break
            
            # Compute finished_mask FIRST to avoid target overflow
            session_lengths = self.session_ends[slot_session] - self.session_starts[slot_session]
            finished_mask = (slot_pos >= session_lengths - 1)
            
            # Build mask of active (non-finished, non-exhausted) slots
            active_mask = (slot_session >= 0) & ~finished_mask
            if not active_mask.any():
                # All remaining slots are either exhausted or at last item
                # Replace finished sessions and continue
                if finished_mask.any():
                    finished_slots = np.where(finished_mask & (slot_session >= 0))[0]
                    for slot in finished_slots:
                        if next_session_idx < self.n_sessions:
                            slot_session[slot] = next_session_idx
                            slot_pos[slot] = 0
                            next_session_idx += 1
                            for layer_h in hidden:
                                layer_h[slot] = 0.0
                        else:
                            slot_session[slot] = -1
                continue
            
            # Get indices only for active slots (safe: no overflow)
            active_slots = np.where(active_mask)[0]
            current_indices = self.session_starts[slot_session[active_slots]] + slot_pos[active_slots]
            input_items = self.items[current_indices]
            target_items = self.items[current_indices + 1]  # Safe: active slots have valid targets

            input_idx = torch.tensor(input_items, dtype=torch.long, device=self.device)
            target_idx = torch.tensor(target_items, dtype=torch.long, device=self.device)
            
            # Extract hidden states for active slots only
            active_hidden = [h[active_slots] for h in hidden]
            n_active = len(active_slots)
            
            logits = None
            loss_val = 0.0
            
            if training:
                if neg_sampler is not None and n_neg > 0:
                    # Sampled softmax with random negatives
                    if logq > 0:
                        # With logq correction
                        negative_idx, neg_logq = neg_sampler.sample(n_active, n_neg, return_logq=True)
                        target_logq = neg_sampler.get_logq(target_idx)
                        target_scores, negative_scores, new_active_hidden = model.forward_with_negatives(
                            input_idx, active_hidden, target_idx, negative_idx
                        )
                        loss = sampled_softmax_loss(
                            target_scores, negative_scores,
                            target_logq=logq * target_logq,
                            negative_logq=logq * neg_logq
                        )
                    else:
                        # No logq correction
                        negative_idx = neg_sampler.sample(n_active, n_neg)
                        target_scores, negative_scores, new_active_hidden = model.forward_with_negatives(
                            input_idx, active_hidden, target_idx, negative_idx
                        )
                        loss = sampled_softmax_loss(target_scores, negative_scores)
                elif n_active > 1:
                    # TOP1 loss with in-batch negatives (original GRU4Rec)
                    h, new_active_hidden = model.step(input_idx, active_hidden)
                    
                    # Positive scores: dot product with true next item embedding
                    target_emb = model.embedding(target_idx)  # (B, D)
                    pos_scores = (h * target_emb).sum(dim=1)  # (B,)
                    
                    # Add bias if weight tying
                    if model.constrained_embedding:
                        pos_scores = pos_scores + model.output_bias[target_idx]
                    
                    # In-batch negatives: use other targets in batch as negatives
                    # all_scores[i,j] = score of sample i against target j
                    all_scores = torch.mm(h, target_emb.t())  # (B, B)
                    if model.constrained_embedding:
                        all_scores = all_scores + model.output_bias[target_idx].unsqueeze(0)
                    
                    # Remove diagonal (self-comparisons) to get negatives
                    mask = ~torch.eye(n_active, dtype=torch.bool, device=h.device)
                    neg_scores = all_scores[mask].view(n_active, n_active - 1)  # (B, B-1)
                    
                    loss = top1_loss(pos_scores, neg_scores)
                else:
                    # Fallback for single sample: use full softmax
                    logits, new_active_hidden = model(input_idx, active_hidden)
                    loss = nn.functional.cross_entropy(logits, target_idx)
                
                # Scale loss for accumulation
                (loss / accum_steps).backward()
                loss_val = loss.item()
                
                step_count += 1
                
                # Only step every accum_steps
                if optimizer is not None and step_count % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Write back updated hidden states
                for i, layer_h in enumerate(hidden):
                    layer_h[active_slots] = new_active_hidden[i].detach()
            else:
                # Evaluation: compute logits with full softmax
                with torch.no_grad():
                    logits, new_active_hidden = model(input_idx, active_hidden)
                    # Write back updated hidden states
                    for i, layer_h in enumerate(hidden):
                        layer_h[active_slots] = new_active_hidden[i]

            # Advance position for active slots only
            slot_pos[active_slots] += 1
            
            # Yield results for active slots
            yield input_idx, target_idx, logits, loss_val, active_slots
            
            # Check which active slots are now finished after this step
            new_session_lengths = self.session_ends[slot_session[active_slots]] - self.session_starts[slot_session[active_slots]]
            newly_finished = (slot_pos[active_slots] >= new_session_lengths - 1)
            
            # Replace newly finished sessions
            if newly_finished.any():
                finished_slot_indices = active_slots[newly_finished]
                for slot in finished_slot_indices:
                    if next_session_idx < self.n_sessions:
                        slot_session[slot] = next_session_idx
                        slot_pos[slot] = 0
                        next_session_idx += 1
                        for layer_h in hidden:
                            layer_h[slot] = 0.0
                    else:
                        slot_session[slot] = -1


def evaluate(model, data, item_to_idx, batch_size, device, cutoff=20):
    """Evaluate model using session-parallel processing."""
    model.eval()
    iterator = SessionParallelIterator(data, batch_size, item_to_idx, device=device)
    
    total_recall = 0.0
    total_mrr = 0.0
    n_samples = 0
    
    # Iterator handles hidden states internally
    # Yields: (input_idx, target_idx, logits, loss_val, active_slots)
    for input_idx, target_idx, logits, _, _ in iterator(model, training=False):
        B = target_idx.shape[0]
        
        _, top_k = torch.topk(logits, k=cutoff, dim=1)
        hits = (top_k == target_idx.unsqueeze(1))
        
        hit_any = hits.any(dim=1).float()
        total_recall += hit_any.sum().item()
        
        hit_positions = hits.float().argmax(dim=1) + 1
        mrr_contributions = torch.where(
            hit_any.bool(),
            1.0 / hit_positions.float(),
            torch.zeros_like(hit_positions.float())
        )
        total_mrr += mrr_contributions.sum().item()
        n_samples += B

    recall = total_recall / n_samples if n_samples > 0 else 0.0
    mrr = total_mrr / n_samples if n_samples > 0 else 0.0
    return recall, mrr


# =============================================================================
# MAIN (FIXED: use Adagrad like official GRU4Rec, or lower Adam LR)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TRUE GRU4Rec Session-Parallel Batching')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=48, help='Batch size (parallel sessions)')
    parser.add_argument('--hidden-size', type=int, default=480, help='GRU hidden size')
    parser.add_argument('--embedding-dim', type=int, default=0, help='Embedding dim (0=same as hidden)')
    parser.add_argument('--n-layers', type=int, default=1, help='Number of GRU layers')
    parser.add_argument('--learning-rate', type=float, default=0.07, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adagrad', choices=['adagrad', 'adam'], help='Optimizer')
    parser.add_argument('--dropout-embed', type=float, default=0.0, help='Embedding dropout')
    parser.add_argument('--dropout-hidden', type=float, default=0.2, help='Hidden dropout')
    parser.add_argument('--n-samples', type=int, default=0, help='Negative samples (0=TOP1 in-batch negatives)')
    parser.add_argument('--sample-alpha', type=float, default=0.75, help='Negative sampling alpha')
    parser.add_argument('--logq', type=float, default=0.0, help='logq correction weight (0=none, 1.0=full). Only used with --n-samples > 0')
    parser.add_argument('--constrained-embedding', action='store_true', default=True, help='Weight tying')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test-size', type=int, default=None, help='Limit sessions for testing')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("TRUE GRU4REC SESSION-PARALLEL BATCHING")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size} parallel sessions")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Negative samples: {args.n_samples}")
    if args.n_samples > 0:
        print(f"Sample alpha: {args.sample_alpha}")
        print(f"logq correction: {args.logq}")
    print(f"Constrained embedding: {args.constrained_embedding}")
    print("="*70 + "\n")
    
    # Load data (use abspath to handle different working directories)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'input_data', 'yoochoose-data'))
    train_path = os.path.join(base_path, 'yoochoose_train_full.dat')
    test_path = os.path.join(base_path, 'yoochoose_test.dat')
    
    print("[1/4] Loading data...")
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    # Create unified vocabulary
    all_items = pd.concat([train_data['item_id'], test_data['item_id']]).unique()
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    n_items = len(item_to_idx)
    print(f"[1/4] Vocabulary: {n_items} items")
    
    # Compute item counts for negative sampling
    item_counts = train_data['item_id'].value_counts().to_dict()
    item_counts = {item_to_idx[k]: v for k, v in item_counts.items() if k in item_to_idx}
    
    # Optionally limit data
    if args.test_size:
        sessions = train_data['session_id'].unique()[:args.test_size]
        train_data = train_data[train_data['session_id'].isin(sessions)]
        print(f"[1/4] Limited to {args.test_size} sessions")
    
    # Create model
    print("[2/4] Creating model...")
    model = SessionGRU(
        n_items=n_items,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        dropout_embed=args.dropout_embed,
        dropout_hidden=args.dropout_hidden,
        constrained_embedding=args.constrained_embedding
    ).to(args.device)
    
    # Create optimizer
    if args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    else:
        # Cap learning rate at 0.001 for Adam
        lr = args.learning_rate if args.learning_rate < 0.01 else 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create negative sampler (only if using sampled softmax)
    neg_sampler = None
    if args.n_samples > 0:
        neg_sampler = NegativeSampler(item_counts, n_items, alpha=args.sample_alpha, device=args.device)
        print(f"[2/4] Using sampled softmax with {args.n_samples} negative samples")
    else:
        print(f"[2/4] Using TOP1 loss with in-batch negatives (original GRU4Rec)")
    
    # Training loop
    print("[3/4] Training...\n")
    
    # Gradient accumulation steps
    ACCUM_STEPS = 1 if args.n_samples == 0 else 50
    
    for epoch in range(1, args.epochs + 1):
        print(f"--- Epoch {epoch}/{args.epochs} ---")
        
        iterator = SessionParallelIterator(
            train_data, args.batch_size, item_to_idx, device=args.device
        )
        
        model.train()
        total_loss = 0.0
        n_steps = 0
        start_time = time.time()
        
        for input_idx, target_idx, _, loss_val, _ in iterator(
            model, optimizer, training=True, 
            neg_sampler=neg_sampler, n_neg=args.n_samples,
            accum_steps=ACCUM_STEPS, logq=args.logq
        ):
            # Accumulate loss
            total_loss += loss_val
            n_steps += 1
            
            if n_steps % 10000 == 0:
                elapsed = time.time() - start_time
                speed = n_steps * args.batch_size / elapsed
                print(f"  Step {n_steps}, Loss: {total_loss/n_steps:.4f}, Speed: {speed:.0f} samples/s")
        
        # Final gradient step if not aligned
        if n_steps % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} completed: Loss={total_loss/n_steps:.4f}, Time={elapsed:.1f}s\n")
    
    # Evaluation
    print("[4/4] Evaluating...")
    recall, mrr = evaluate(model, test_data, item_to_idx, args.batch_size, args.device)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Recall@20: {recall:.6f}")
    print(f"MRR@20:    {mrr:.6f}")
    print("="*70 + "\n")
    
    # Save model
    output_path = os.path.join(os.path.dirname(__file__), 'gru4rec_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'item_to_idx': item_to_idx,
        'n_items': n_items,
        'hidden_size': args.hidden_size,
        'embedding_dim': args.embedding_dim,
        'n_layers': args.n_layers,
        'constrained_embedding': args.constrained_embedding,
        'recall': recall,
        'mrr': mrr
    }, output_path)
    print(f"Model saved to {output_path}")


if __name__ == '__main__':
    main()
