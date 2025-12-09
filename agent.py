"""
DQN Training Agent for pixel-based reinforcement learning.

This module implements a Deep Q-Network trainer with support for:
- Double DQN
- Data augmentation (DrQ)
- SVEA stabilization
- Mixed precision training
- Experience replay with N-step returns
"""

import os
import copy
import time
import random
import numpy as np
import torch
from collections import deque
from kornia import augmentation as aug
from torch.utils.tensorboard import SummaryWriter


class DQNAgent:
    """
    Deep Q-Network training agent with modern improvements.
    
    Supports Double DQN, data augmentation, noisy networks,
    and mixed precision training for efficient learning.
    
    Args:
        env: Gym environment (pixel observations recommended)
        replay_buffer: Buffer instance for experience storage
        frame_stack: Number of frames to stack as observation
        discount: Discount factor γ
        epsilon: Initial exploration rate (0 for noisy networks)
        epsilon_schedule: Function(eps, step) -> new_eps for decay
        target_sync_freq: Learning steps between target network syncs
        update_freq: Environment steps between network updates
        network: Q-network model (AbstractFullyConnected subclass)
        learning_rate: Optimizer learning rate
        decay_lr: Whether to decay learning rate over training
        loss_fn: Loss function for Q-learning
        batch_size: Minibatch size for updates
        device: Training device ('cuda' required for mixed precision)
        double_q: Use Double DQN update rule
        augment_data: Use DrQ data augmentation
        use_svea: Use SVEA for stabilization (requires augment_data)
        reset_interval: Steps between weight resets (0 to disable)
        num_targets: Number of target networks for averaging
        experiment_name: Suffix for save directory
        disable_saving: Skip saving logs and checkpoints
    """
    
    def __init__(
        self,
        env,
        replay_buffer,
        frame_stack,
        discount,
        epsilon,
        epsilon_schedule,
        target_sync_freq,
        update_freq,
        network,
        learning_rate,
        decay_lr,
        loss_fn,
        batch_size,
        device,
        double_q=True,
        augment_data=True,
        use_svea=True,
        reset_interval=0,
        num_targets=1,
        experiment_name='',
        disable_saving=False
    ):
        # Environment and buffer
        self.env = env
        self.replay_buffer = replay_buffer
        
        # Training parameters
        if frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        self.frame_stack = frame_stack
        self.discount = discount
        self.eps = epsilon
        self.epsilon_schedule = epsilon_schedule
        self.target_sync_freq = target_sync_freq
        self.update_freq = max(1, int(update_freq))
        self.updates_per_step = max(1, int(1.0 / update_freq))
        
        # Networks
        if num_targets <= 0:
            raise ValueError("num_targets must be positive")
        self.network = network.to(device)
        self.target_networks = [copy.deepcopy(self.network) for _ in range(num_targets)]
        
        # Optimizer with learning rate decay support
        self.initial_lr = learning_rate
        self.minimum_lr = learning_rate * 0.625
        self.decay_lr = decay_lr
        self.optimizer = torch.optim.NAdam(
            self.network.parameters(),
            lr=learning_rate,
            eps=0.005 / batch_size
        )
        
        # Set networks to eval mode, freeze targets
        self.network.eval()
        for target in self.target_networks:
            target.eval()
            for p in target.parameters():
                p.requires_grad = False
        
        # Loss function
        self.loss_fn = loss_fn
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(device)
        if hasattr(self.loss_fn, 'reduction'):
            self.loss_fn.reduction = 'none'
        
        self.batch_size = batch_size
        
        if device != 'cuda':
            raise ValueError("CUDA required for mixed precision training")
        self.device = device
        self.grad_scaler = torch.cuda.amp.grad_scaler.GradScaler()
        
        # Algorithm variants
        self.use_double_q = double_q
        if augment_data:
            if len(self.env.observation_space.shape) != 3:
                raise ValueError("Data augmentation requires image observations")
        
        obs_shape = self.env.observation_space.shape[1:]
        pad_size = tuple(np.array(obs_shape, dtype=int) // 20)
        self.augment = aug.RandomCrop(
            size=obs_shape,
            padding=pad_size,
            padding_mode='replicate'
        ).to(device) if augment_data else None
        
        self.use_svea = use_svea
        self.reset_interval = reset_interval
        if use_svea and not augment_data:
            raise ValueError("SVEA requires data augmentation to be enabled")
        
        # Counters
        self.env_steps = 0
        self.gradient_steps = 0
        self.target_updates = 0
        self._steps_since_sync = 0
        
        # Logging
        self.disable_saving = disable_saving
        timestamp = str(int(time.time()))
        save_dir = f'./saved/{timestamp}{experiment_name}'
        self.save_dir = save_dir if save_dir.endswith('/') else f'{save_dir}/'
        
        if not disable_saving:
            print(f'Experiment directory: {self.save_dir}')
            os.makedirs(self.save_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(self.save_dir + 'tensorboard/')
        
        # Warmup to avoid slow first inference
        self._warmup_networks()
    
    # =========================================================================
    # Observation Processing
    # =========================================================================
    
    @staticmethod
    def _stack_frames(frames):
        """Convert frame deque to tuple for storage."""
        return tuple(frames)
    
    @staticmethod
    def _normalize_obs(obs):
        """Normalize observations from [0, 255] to [-1, 1]."""
        obs /= 127.5
        obs -= 1.0
        return obs
    
    @torch.no_grad()
    def _preprocess_batch(self, obs, apply_augment=True, include_original=False):
        """Preprocess observation batch for training."""
        obs = torch.as_tensor(obs, device=self.device)
        if len(obs.shape) != 4:
            return obs
        
        obs = self._normalize_obs(obs)
        
        if self.augment and apply_augment:
            scale = torch.randn((self.batch_size, 1, 1, 1), device=self.device)
            scale = torch.clip_(scale, -2, 2) * 0.05 + 1.0
            augmented = torch.vstack([obs * scale, self.augment(obs)])
            augmented = torch.clip_(augmented, -1, 1)
            if include_original:
                augmented = torch.vstack([augmented, obs])
            return augmented
        return obs
    
    def _warmup_networks(self):
        """Run dummy forward passes to initialize CUDA kernels."""
        c, *spatial = self.env.observation_space.shape
        dummy_input = torch.rand(
            (self.batch_size, self.frame_stack * c) + tuple(spatial),
            device=self.device
        )
        for _ in range(3):
            with torch.amp.autocast(self.device):
                self.network(dummy_input).detach().cpu().numpy()
                for target in self.target_networks:
                    target(dummy_input).detach().cpu().numpy()
    
    # =========================================================================
    # Target Network Management
    # =========================================================================
    
    def _sync_target(self, target_idx):
        """Copy online network weights to specified target network."""
        if target_idx < len(self.target_networks):
            self.target_networks[target_idx].load_state_dict(self.network.state_dict())
            self.target_networks[target_idx].eval()
            if target_idx == 0:
                self.target_updates += 1
                self._steps_since_sync = 0
                if self.target_sync_freq > 500:
                    print(f'Target network synced (#{self.target_updates})')
    
    @torch.no_grad()
    def _compute_targets(self, next_obs, rewards, dones):
        """Compute TD targets for Q-learning update."""
        with torch.amp.autocast(self.device):
            next_obs = self._preprocess_batch(next_obs, apply_augment=not self.use_svea, include_original=False)
            rewards = torch.as_tensor(rewards, device=self.device)
            dones = torch.as_tensor(dones, device=self.device)
            
            target_q = self.target_networks[0](next_obs)
            for target in self.target_networks[1:]:
                target_q += target(next_obs)
            if len(self.target_networks) > 1:
                target_q /= len(self.target_networks)
            
            if self.use_double_q:
                with torch.inference_mode():
                    next_actions = self.network(next_obs, adv_only=True)
                    next_actions = torch.argmax(next_actions, dim=-1, keepdim=True)
                max_q = torch.gather(target_q, -1, next_actions)
            else:
                max_q, _ = target_q.max(dim=-1, keepdim=True)
            
            if self.augment and not self.use_svea:
                max_q = (max_q[:self.batch_size] + max_q[self.batch_size:]) / 2.0
            
            targets = rewards + self.discount * max_q * (1.0 - dones)
        return targets.detach()
    
    # =========================================================================
    # Action Selection
    # =========================================================================
    
    @torch.inference_mode()
    def select_action(self, obs):
        """Select action with highest Q-value (greedy)."""
        with torch.amp.autocast(self.device):
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            if len(obs.shape) == 4:
                self._normalize_obs(obs)
            q_values = self.network(obs, adv_only=True).cpu().numpy()[0]
        return np.argmax(q_values)
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    def run_episode(self, explore_randomly=False, save_transitions=False):
        """Execute one episode and learn from experience."""
        cache_dir = self.save_dir + 'episode_cache/'
        if save_transitions:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = f'{int(time.time())}.npz'
            i = 0
            while os.path.exists(cache_dir + cache_file):
                cache_file = f'{int(time.time())}_{i}.npz'
                i += 1
            print(f'Caching transitions to: {os.path.abspath(cache_dir + cache_file)}')
        
        if self.decay_lr and not explore_randomly and self.target_updates > 0:
            lr_decay = (self.initial_lr - self.minimum_lr) / 300.0
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(self.minimum_lr, param_group['lr'] - lr_decay)
        
        initial_obs, _ = self.env.reset()
        frame_buffer = deque((initial_obs for _ in range(self.frame_stack)), maxlen=self.frame_stack)
        
        obs_history = [initial_obs]
        action_history, reward_history, done_history = [], [], []
        
        episode_reward = 0.0
        total_loss = 0.0
        update_count = 0
        current_frames = self._stack_frames(frame_buffer)
        
        while True:
            if explore_randomly or self.eps > random.random():
                action = self.env.action_space.sample()
            else:
                model_input = np.concatenate(current_frames, dtype=np.float32)
                action = self.select_action(model_input)
            
            next_obs, reward, done, _, _ = self.env.step(action)
            episode_reward += reward
            self.env_steps += 1
            
            if save_transitions:
                obs_history.append(next_obs)
                action_history.append(action)
                reward_history.append(reward)
                done_history.append(done)
            
            frame_buffer.append(next_obs)
            next_frames = self._stack_frames(frame_buffer)
            self.replay_buffer.add(current_frames, action, reward, done, next_frames)
            
            if self.reset_interval and self.env_steps % self.reset_interval == 0:
                print('Resetting network parameters')
                self.network.reset_params()
                for i in range(len(self.target_networks)):
                    self._sync_target(i)
            
            if not explore_randomly:
                self.eps = self.epsilon_schedule(self.eps, self.env_steps)
                if len(self.replay_buffer) > self.batch_size and self.env_steps % self.update_freq == 0:
                    for _ in range(self.updates_per_step):
                        total_loss += self._update_network()
                        update_count += 1
            
            current_frames = next_frames
            if done:
                break
        
        if not explore_randomly:
            self.replay_buffer.step()
        
        if save_transitions:
            self._save_episode(obs_history, action_history, reward_history, done_history, cache_dir + cache_file)
        
        avg_loss = total_loss / update_count if update_count > 0 else 0.0
        return episode_reward, avg_loss, self.optimizer.param_groups[0]['lr']
    
    def run_multiple_episodes(self, num_episodes, **kwargs):
        """Run multiple episodes with given settings."""
        for _ in range(num_episodes):
            self.run_episode(**kwargs)
    
    # =========================================================================
    # Network Update
    # =========================================================================
    
    def _update_network(self):
        """Perform one gradient update on the Q-network."""
        if self.replay_buffer.prioritized:
            (obs, actions, rewards, next_obs, dones), indices = self.replay_buffer.prioritized_sample(self.batch_size)
        else:
            obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
            indices = None
        
        self.network.reset_noise()
        for target in self.target_networks:
            target.reset_noise()
        
        targets = self._compute_targets(next_obs, rewards, dones)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        
        self.network.train()
        with torch.amp.autocast(self.device):
            obs = self._preprocess_batch(obs, apply_augment=True, include_original=self.use_svea)
            obs.requires_grad = True
            self.optimizer.zero_grad(set_to_none=True)
            q_values = self.network(obs)
            
            if self.augment:
                if self.use_svea:
                    q_values = (q_values[:self.batch_size] + q_values[self.batch_size:2*self.batch_size] + q_values[2*self.batch_size:]) / 3.0
                else:
                    q_values = (q_values[:self.batch_size] + q_values[self.batch_size:]) / 2.0
            
            q_selected = torch.gather(q_values, -1, actions)
            loss = self.loss_fn(q_selected, targets)
            
            if self.replay_buffer.prioritized:
                with torch.no_grad():
                    td_errors = (q_selected - targets).abs().cpu().numpy().flatten()
                    weights = self.replay_buffer.update_priority(td_errors, indices)
                weights = torch.as_tensor(weights, device=self.device)
                loss *= weights.reshape(loss.shape)
            loss = loss.mean()
        
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.network.eval()
        
        with torch.no_grad():
            loss_value = float(loss.detach().cpu().numpy())
            self._steps_since_sync += 1
            self.gradient_steps += 1
            self._sync_target(self._steps_since_sync % self.target_sync_freq)
        return loss_value
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    def evaluate_policy(self):
        """Run one episode with greedy policy (no exploration)."""
        self.network.noise_mode(False)
        initial_obs, _ = self.env.reset()
        frame_buffer = deque((initial_obs for _ in range(self.frame_stack)), maxlen=self.frame_stack)
        
        episode_reward = 0.0
        while True:
            frames = tuple(frame_buffer)
            model_input = np.concatenate(frames, dtype=np.float32)
            action = self.select_action(model_input)
            next_obs, reward, done, _, _ = self.env.step(action)
            episode_reward += reward
            frame_buffer.append(next_obs)
            if done:
                break
        
        self.network.noise_mode(True)
        print(f'Evaluation reward: {episode_reward:.2f}')
        return episode_reward
    
    # =========================================================================
    # Data Management
    # =========================================================================
    
    @staticmethod
    def _save_episode(obs_list, action_list, reward_list, done_list, filepath):
        """Save episode transitions to compressed numpy file."""
        assert len(obs_list) - 1 == len(action_list) == len(reward_list) == len(done_list)
        assert not os.path.exists(filepath)
        obs_arr = np.array(obs_list, dtype=obs_list[0].dtype)
        action_arr = np.array(action_list, dtype=np.uint8 if max(action_list) < 256 else np.uint32)
        reward_arr = np.array(reward_list, dtype=np.float32)
        done_arr = np.array(done_list, dtype=np.bool_)
        np.savez_compressed(filepath, o=obs_arr, a=action_arr, r=reward_arr, d=done_arr)
    
    def load_exploration_data(self, data_dir='./explorations/'):
        """Load pre-collected exploration data into replay buffer."""
        data_dir = data_dir if data_dir.endswith('/') else f'{data_dir}/'
        for filename in os.listdir(data_dir):
            if not filename.endswith('.npz'):
                continue
            filepath = data_dir + filename
            print(f'Loading: {os.path.abspath(filepath)}')
            data = np.load(filepath)
            obs_arr, action_arr, reward_arr, done_arr = data['o'], data['a'], data['r'], data['d']
            if obs_arr[0].shape != self.env.observation_space.shape:
                raise ValueError(f"Observation shape mismatch in {filename}")
            frame_buffer = deque((obs_arr[0] for _ in range(self.frame_stack)), maxlen=self.frame_stack)
            current_frames = self._stack_frames(frame_buffer)
            for obs, act, rew, done in zip(obs_arr[1:], action_arr, reward_arr, done_arr):
                prev_frames = current_frames
                frame_buffer.append(obs)
                current_frames = self._stack_frames(frame_buffer)
                self.replay_buffer.add(prev_frames, act, rew, done, current_frames)
        print(f'Loaded {len(self.replay_buffer)} transitions into buffer')
    
    def collect_exploration_data(self, num_episodes, save_dir='./explorations/'):
        """Collect random exploration data and save to disk."""
        save_dir = save_dir if save_dir.endswith('/') else f'{save_dir}/'
        os.makedirs(save_dir, exist_ok=True)
        for i in range(num_episodes):
            filepath = f'{save_dir}{i}.npz'
            if os.path.exists(filepath):
                print(f'Skipping existing: {os.path.abspath(filepath)}')
                continue
            obs, _ = self.env.reset()
            obs_list = [obs]
            action_list, reward_list, done_list = [], [], []
            while True:
                action = self.env.action_space.sample()
                next_obs, reward, done, _, _ = self.env.step(action)
                obs_list.append(next_obs)
                action_list.append(action)
                reward_list.append(reward)
                done_list.append(done)
                if done:
                    break
            self._save_episode(obs_list, action_list, reward_list, done_list, filepath)
            print(f'Saved exploration: {os.path.abspath(filepath)}')
            print(f'Episode reward: {sum(reward_list):.2f}')
    
    # =========================================================================
    # Checkpointing
    # =========================================================================
    
    def save_checkpoint(self, name='', online_only=False):
        """Save model checkpoint."""
        if self.disable_saving:
            return
        torch.save(self.network.state_dict(), f'{self.save_dir}{name}online.pt')
        if online_only:
            return
        for i, target in enumerate(self.target_networks):
            torch.save(target.state_dict(), f'{self.save_dir}{name}target{i}.pt')
        torch.save(self.optimizer.state_dict(), f'{self.save_dir}{name}optimizer.pt')
    
    def log_metrics(self, metrics, step):
        """Log metrics to TensorBoard."""
        if not self.disable_saving:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)
    
    # =========================================================================
    # Backward Compatibility Properties
    # =========================================================================
    
    @property
    def steps(self):
        return self.env_steps
    
    @steps.setter
    def steps(self, value):
        self.env_steps = value
    
    @property
    def learn_steps(self):
        return self.gradient_steps
    
    @property
    def model(self):
        return self.network
    
    @property
    def n_frames(self):
        return self.frame_stack
    
    @property
    def gamma(self):
        return self.discount


# Backward compatibility alias
Trainer = DQNAgent

# Method aliases
DQNAgent.evaluate = DQNAgent.evaluate_policy
DQNAgent.save_models = DQNAgent.save_checkpoint
DQNAgent.log = DQNAgent.log_metrics
DQNAgent.save_explorations = DQNAgent.collect_exploration_data
DQNAgent.load_explorations = DQNAgent.load_exploration_data
DQNAgent.get_action = DQNAgent.select_action
DQNAgent.learn = DQNAgent._update_network

