"""
Experience replay buffers for off-policy reinforcement learning.

This module provides replay buffer implementations including
standard uniform sampling and N-step return computation.
"""

import random
import numpy as np
from collections import deque

from priority_tree import PrioritySumTree


# =============================================================================
# Base Replay Buffer
# =============================================================================

class ReplayMemory:
    """
    Experience replay buffer with uniform random sampling.
    
    Stores (state, action, reward, next_state, done) transitions
    and supports optional prioritized experience replay.
    
    Args:
        capacity: Maximum number of transitions to store
        priority_config: Dict with PER settings, or None for uniform sampling
    """
    
    def __init__(self, capacity, priority_config=None, *args, **kwargs):
        if capacity <= 0:
            raise ValueError("Buffer capacity must be positive")
        
        self.capacity = capacity
        self.use_priority = priority_config is not None
        
        if self.use_priority:
            self._storage = PrioritySumTree(maxlen=capacity, **priority_config)
        else:
            self._storage = deque(maxlen=capacity)
        
        self._pending = []  # Temporary storage for incomplete transitions
    
    @staticmethod
    def _batch_to_arrays(batch):
        """Convert batch of transitions to numpy arrays."""
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for s, a, r, s_next, d in batch:
            states.append(np.concatenate(s))
            actions.append(a)
            rewards.append(r)
            next_states.append(np.concatenate(s_next))
            dones.append(d)
        
        return (
            np.array(states, dtype=np.float32, copy=True),
            np.array(actions, dtype=np.int64)[:, np.newaxis],
            np.array(rewards, dtype=np.float32)[:, np.newaxis],
            np.array(next_states, dtype=np.float32, copy=True),
            np.array(dones, dtype=np.float32)[:, np.newaxis]
        )
    
    @property
    def is_full(self):
        """Check if buffer has reached capacity."""
        return len(self._storage) == self.capacity
    
    def store(self, state, action, reward, done, next_state):
        """
        Store a transition in the buffer.
        
        Note: Uses delayed storage to ensure next_state alignment.
        """
        self._pending.append((state, action, reward, done))
        
        if len(self._pending) == 2:
            prev_s, prev_a, prev_r, prev_d = self._pending.pop(0)
            self._storage.append((prev_s, prev_a, prev_r, state, prev_d))
            
            if done:
                # Episode ended - flush remaining
                self._storage.append((state, action, reward, next_state, done))
                self._pending.clear()
    
    def sample_batch(self, batch_size):
        """Sample a batch of transitions uniformly at random."""
        batch = random.sample(list(self._storage), batch_size)
        return self._batch_to_arrays(batch)
    
    def sample_prioritized(self, batch_size):
        """Sample a batch using prioritized experience replay."""
        batch, indices = self._storage.sample(batch_size)
        return self._batch_to_arrays(batch), indices
    
    def update_priorities(self, td_errors, indices):
        """Update priorities based on TD errors."""
        importance_weights = []
        for error, idx in zip(td_errors, indices):
            weight = self._storage.update_prio(error, idx)
            importance_weights.append(weight)
        
        weights = np.array(importance_weights, dtype=np.float32)
        return weights / np.max(weights)  # Normalize
    
    def end_episode(self):
        """Called at episode end for any cleanup (e.g., beta annealing)."""
        if self.use_priority:
            self._storage.step_beta()
    
    def __len__(self):
        return len(self._storage)
    
    def __repr__(self):
        return f"ReplayMemory(size={len(self)}, capacity={self.capacity})"


# =============================================================================
# N-Step Return Buffer
# =============================================================================

class NStepReplayMemory(ReplayMemory):
    """
    Replay buffer with N-step return computation.
    
    Computes multi-step bootstrapped returns:
        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^n * V(s_{t+n})
    
    Args:
        capacity: Maximum number of transitions
        n_steps: Number of steps for return computation
        discount: Discount factor γ
        priority_config: Optional PER configuration
    """
    
    def __init__(self, capacity, n_steps=5, discount=0.99, priority_config=None):
        super().__init__(capacity, priority_config=priority_config)
        self.n_steps = n_steps
        self.discount = discount
    
    def _flush_nstep(self, bootstrap_state, terminal):
        """Compute and store N-step return for oldest pending transition."""
        state, action, reward, _ = self._pending.pop(0)
        
        # Accumulate discounted rewards
        gamma_power = self.discount
        for pending_transition in self._pending:
            reward += gamma_power * pending_transition[2]
            gamma_power *= self.discount
        
        self._storage.append((state, action, reward, bootstrap_state, terminal))
    
    def store(self, state, action, reward, done, next_state):
        """Store transition with N-step return computation."""
        self._pending.append((state, action, reward, done))
        
        if done:
            # Episode ended - flush all with proper bootstrapping
            self._flush_nstep(state, False)  # First one bootstraps from current
            while len(self._pending) > 0:
                self._flush_nstep(next_state, done)
        elif len(self._pending) > self.n_steps:
            # Buffer full - compute N-step return
            self._flush_nstep(state, done)


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

Buffer = ReplayMemory
MultistepBuffer = NStepReplayMemory

ReplayMemory.add = ReplayMemory.store
ReplayMemory.sample = ReplayMemory.sample_batch
ReplayMemory.prioritized_sample = ReplayMemory.sample_prioritized
ReplayMemory.update_priority = ReplayMemory.update_priorities
ReplayMemory.step = ReplayMemory.end_episode
ReplayMemory.prioritized = property(lambda self: self.use_priority)

NStepReplayMemory.add = NStepReplayMemory.store
NStepReplayMemory.n = property(lambda self: self.n_steps)
NStepReplayMemory.gamma = property(lambda self: self.discount)

