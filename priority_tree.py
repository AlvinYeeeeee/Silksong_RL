"""
Sum Tree data structure for Prioritized Experience Replay.

Provides O(log n) sampling proportional to priority values,
enabling efficient prioritized replay buffer implementation.
"""

import random
import numpy as np


class PrioritySumTree:
    """
    Binary sum tree for efficient priority-based sampling.
    
    Each leaf stores a transition and its priority. Internal nodes
    store the sum of priorities in their subtrees, enabling O(log n)
    proportional sampling.
    
    Args:
        maxlen: Maximum capacity (number of leaves)
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        beta: Importance sampling correction (annealed from initial to 1)
        beta_anneal: Amount to increase beta each episode
    """
    
    def __init__(self, maxlen, alpha=0.6, beta=0.4, beta_anneal=0.0):
        self.maxlen = maxlen
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_anneal
        
        # Data storage
        self._data = [None] * maxlen
        
        # Priority tree (2n-1 nodes for n leaves)
        self._tree = np.zeros(maxlen * 2 - 1, dtype=np.float64)
        
        self._size = 0
        self._write_idx = 0
        self._max_priority = 1.0
    
    def _traverse(self, value):
        """Traverse tree to find leaf index for given cumulative value."""
        node = 0
        while True:
            left = 2 * node + 1
            if left >= len(self._tree):
                break  # Reached leaf level
            
            if value <= self._tree[left]:
                node = left
            else:
                value -= self._tree[left]
                node = left + 1
        
        # Convert tree index to leaf index
        return node - self.maxlen + 1
    
    def update_prio(self, priority, leaf_idx, return_weight=True):
        """
        Update priority for a leaf and propagate changes up the tree.
        
        Args:
            priority: New priority value (before alpha transform)
            leaf_idx: Index of the leaf to update
            return_weight: Whether to return importance sampling weight
            
        Returns:
            Importance sampling weight if return_weight=True
        """
        if priority > self._max_priority:
            self._max_priority = priority
        
        # Apply priority exponent
        transformed_priority = (priority + 1e-7) ** self.alpha
        
        tree_idx = self.maxlen - 1 + leaf_idx
        old_priority = self._tree[tree_idx]
        total_priority = self._tree[0]
        delta = transformed_priority - old_priority
        
        # Propagate change up to root
        while tree_idx >= 0:
            self._tree[tree_idx] += delta
            tree_idx = (tree_idx - 1) // 2
        
        if return_weight:
            # Compute importance sampling weight
            prob = old_priority / total_priority
            weight = (1.0 / (self._size * prob)) ** self.beta
            return weight
    
    def append(self, element):
        """
        Add new element with maximum priority.
        
        Returns:
            Index where element was stored
        """
        idx = self._write_idx
        self._data[idx] = element
        self.update_prio(self._max_priority, idx, return_weight=False)
        
        self._write_idx = (idx + 1) % self.maxlen
        if self._size < self.maxlen:
            self._size += 1
        
        return idx
    
    def sample(self, batch_size):
        """
        Sample batch proportional to priorities.
        
        Uses stratified sampling: divides total priority into k segments
        and samples one item from each segment.
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            Tuple of (elements, indices)
        """
        if batch_size > self._size:
            raise ValueError(f"Requested {batch_size} samples but only have {self._size}")
        
        segment_size = self._tree[0] / batch_size
        sampled_data = []
        sampled_indices = []
        
        for i in range(batch_size):
            # Sample uniformly within segment
            low = i * segment_size
            high = (i + 1) * segment_size
            value = random.uniform(low, high)
            
            idx = self._traverse(value)
            
            # Handle rare floating-point edge cases
            while self._data[idx] is None:
                value = random.uniform(low, high)
                idx = self._traverse(value)
            
            sampled_indices.append(idx)
            sampled_data.append(self._data[idx])
        
        return sampled_data, sampled_indices
    
    def step_beta(self):
        """Anneal beta towards 1 (full importance sampling correction)."""
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def __len__(self):
        return self._size
    
    def __repr__(self):
        return f"PrioritySumTree(size={self._size}, capacity={self.maxlen}, total={self._tree[0]:.2f})"


# Backward compatibility alias
SumTree = PrioritySumTree


if __name__ == '__main__':
    # Quick test
    tree = PrioritySumTree(10)
    for i in range(4):
        idx = tree.append((f'item_{i}',))
        tree.update_prio(random.uniform(0.1, 1.0), idx)
    print(f"Tree: {tree}")
    samples, indices = tree.sample(3)
    print(f"Sampled: {samples}")

