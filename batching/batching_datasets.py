from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import numpy as np
import math

class SessionDataset(Dataset):
    def __init__(self, sessions: List[List[int]], mode: str = 'all', max_seq_len: int = 50, pad_idx: int = 0, inplace_shuffle: bool = False):
        assert mode in ('all', 'last')
        self.sessions = sessions
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.inplace_shuffle = inplace_shuffle
        self.samples: List[Tuple[List[int], int]] = []
        for seq in self.sessions:
            if self.mode == 'all':
                for i in range(1, len(seq)):
                    self.samples.append((seq[:i], int(seq[i])))
            else:
                self.samples.append((seq[:-1], int(seq[-1])))
        if self.inplace_shuffle:
            import random
            random.shuffle(self.samples)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        return seq, int(target)

def collate_fn(batch: List[Tuple[List[int], int]], pad_idx: int = 0, align: str = 'right') -> Dict[str, torch.Tensor]:
    seqs, targets = zip(*batch)
    tensors = [torch.tensor(s, dtype=torch.long) for s in seqs]
    lengths = torch.tensor([t.size(0) for t in tensors], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.full((len(tensors), max_len), fill_value=pad_idx, dtype=torch.long)
    attention_mask = torch.zeros((len(tensors), max_len), dtype=torch.long)
    for i, t in enumerate(tensors):
        l = t.size(0)
        if align == 'right':
            padded[i, max_len - l: max_len] = t
            attention_mask[i, max_len - l: max_len] = 1
        else:
            padded[i, :l] = t
            attention_mask[i, :l] = 1
    return {
        'input_ids': padded,
        'attention_mask': attention_mask,
        'lengths': lengths,
        'targets': torch.tensor(targets, dtype=torch.long)
    }

class SessionParallelDataset(IterableDataset):
    def __init__(self, sessions: List[List[int]], batch_size: int = 512, shuffle: bool = True):
        assert all(len(s) >= 2 for s in sessions), "All sessions must have length >= 2"
        self.sessions = sessions
        self.batch_size = batch_size
        self.shuffle = shuffle
    def _generator(self, sessions: List[List[int]]):
        order = np.arange(len(sessions))
        if self.shuffle:
            np.random.shuffle(order)
        q_ptr = 0
        active = []
        pos = []
        session_idx = []
        while q_ptr < len(order) and len(active) < self.batch_size:
            sidx = order[q_ptr]
            q_ptr += 1
            active.append(sessions[sidx])
            pos.append(0)
            session_idx.append(sidx)
        if len(active) == 0:
            return
        new_mask = np.ones(len(active), dtype=np.bool_)
        while len(active) > 0:
            B = len(active)
            inputs = np.empty(B, dtype=np.int64)
            targets = np.empty(B, dtype=np.int64)
            for i in range(B):
                inputs[i] = active[i][pos[i]]
                targets[i] = active[i][pos[i] + 1]
            batch = {
                'inputs': torch.from_numpy(inputs).long(),
                'targets': torch.from_numpy(targets).long(),
                'session_ids': torch.tensor(session_idx, dtype=torch.long),
                'new_session_mask': torch.from_numpy(new_mask)
            }
            yield batch
            new_mask = np.zeros(B, dtype=np.bool_)
            remove_indices = []
            for i in range(B):
                pos[i] += 1
                if pos[i] >= len(active[i]) - 1:
                    if q_ptr < len(order):
                        sidx = order[q_ptr]
                        q_ptr += 1
                        active[i] = sessions[sidx]
                        pos[i] = 0
                        session_idx[i] = sidx
                        new_mask[i] = True
                    else:
                        remove_indices.append(i)
            if remove_indices:
                for i in reversed(remove_indices):
                    active.pop(i)
                    pos.pop(i)
                    session_idx.pop(i)
                if len(new_mask) > 0:
                    new_mask = new_mask[:len(active)]
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            sessions = self.sessions
        else:
            per_worker = int(math.ceil(len(self.sessions) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.sessions))
            sessions = self.sessions[start:end]
        return self._generator(sessions)
