"""
Training script for the Lace boss fight in Hollow Knight: Silksong.

This script trains a DQN agent to defeat the Lace boss using:
- Pixel-based observations (192x192 grayscale)
- Frame stacking (4 frames)
- Enhanced DQN with dueling architecture and noisy networks
- 10-step returns for faster credit assignment

Usage:
    python run_training.py

Prerequisites:
    1. Silksong running in 1280x720 windowed mode
    2. Player positioned in the Lace boss fight arena
    3. Game window focused and visible
"""

import time
import numpy as np
import torch
import psutil
from collections import deque
from torch.backends import cudnn

# Local modules
import environment
import networks
import agent
import replay


# =============================================================================
# Device Configuration
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    cudnn.benchmark = True
print(f"Training device: {DEVICE}")


# =============================================================================
# Model Creation
# =============================================================================

def create_network(env, num_frames):
    """
    Build the Q-network architecture.
    
    Architecture: CNN Feature Extractor → Dueling Q-Network
    - 5-layer CNN: 4 → 32 → 48 → 96 → 160 → 320 channels
    - Dueling head with noisy linear layers for exploration
    
    Args:
        env: Gym environment for observation/action space info
        num_frames: Number of stacked frames as input
        
    Returns:
        Q-network model on the configured device
    """
    channels, *spatial_dims = env.observation_space.shape
    
    # CNN encoder for visual features
    encoder = networks.ConvEncoder(
        spatial_shape=spatial_dims,
        input_channels=num_frames * channels,
        activation='relu',
        use_spectral_norm=False
    )
    
    # Dueling Q-network head with noisy layers
    q_network = networks.DuelingQNetwork(
        encoder=encoder,
        num_actions=env.action_space.n,
        activation='relu',
        noisy=True,
        use_spectral_norm=False
    )
    
    return q_network.to(DEVICE)


# =============================================================================
# Evaluation with Boss HP Tracking
# =============================================================================

def run_evaluation(dqn_agent, num_frames):
    """
    Evaluate current policy and track boss HP.
    
    Runs one episode with greedy action selection (no exploration)
    and returns both the total reward and final boss HP.
    
    Args:
        dqn_agent: DQN agent instance
        num_frames: Frame stack size
        
    Returns:
        Tuple of (total_reward, final_boss_hp_fraction)
    """
    dqn_agent.network.set_exploration(False)
    env = dqn_agent.env
    
    # Initialize frame stack
    initial_obs, _ = env.reset()
    frame_stack = deque(
        (initial_obs for _ in range(num_frames)),
        maxlen=num_frames
    )
    
    episode_reward = 0.0
    boss_hp = 1.0  # Default to 100%
    
    while True:
        # Prepare stacked observation
        stacked = np.concatenate(tuple(frame_stack), dtype=np.float32)
        action = dqn_agent.select_action(stacked)
        
        obs_next, reward, done, _, info = env.step(action)
        episode_reward += reward
        frame_stack.append(obs_next)
        
        # Track boss HP from environment info
        if 'enemy_hp' in info:
            boss_hp = info['enemy_hp']
        
        if done:
            break
    
    dqn_agent.network.set_exploration(True)
    print(f'Evaluation: reward={episode_reward:.2f}, boss_hp={boss_hp*100:.1f}%')
    
    return episode_reward, boss_hp


# =============================================================================
# Main Training Loop
# =============================================================================

def train_agent(dqn_agent, num_frames):
    """
    Execute the full training procedure.
    
    Training consists of two phases:
    1. Exploration: 75 episodes of random play to populate replay buffer
    2. Learning: 550 episodes of DQN training with periodic evaluation
    
    Checkpoints saved:
    - 'best': Highest evaluation reward
    - 'besthp': Lowest average boss HP (most damage dealt)
    - 'besttrain': Highest training reward
    - 'latest': Most recent model
    - 'final': End of training
    
    Args:
        dqn_agent: DQN agent to train
        num_frames: Frame stack size for evaluation
    """
    print('=' * 60)
    print('Starting Training')
    print('=' * 60)
    print('\n⚠️  Switch to the game window now!')
    print('   Make sure you are IN THE BOSS FIGHT (not waiting screen)\n')
    
    for countdown in range(5, 0, -1):
        print(f'   Starting in {countdown}...')
        time.sleep(1)
    print('   GO!\n')
    
    # Phase 1: Collect random exploration data
    print('Phase 1: Exploration (random actions)')
    print('-' * 40)
    dqn_agent.collect_exploration_data(75)
    dqn_agent.load_exploration_data()
    
    # Initial gradient step to initialize optimizer
    dqn_agent._update_network()
    
    # Tracking variables
    best_eval_reward = float('-inf')
    best_train_reward = float('-inf')
    best_boss_hp = float('inf')
    recent_boss_hp = []
    
    # Phase 2: DQN Training
    print('\nPhase 2: DQN Training')
    print('-' * 40)
    
    for episode in range(1, 550):
        print(f'\nEpisode {episode}')
        
        # Train one episode
        reward, loss, lr = dqn_agent.run_episode()
        
        # Save best training model (after initial warmup)
        if reward > best_train_reward and dqn_agent.eps < 0.11:
            print('  → New best training model!')
            best_train_reward = reward
            dqn_agent.save_checkpoint('besttrain', online_only=True)
        
        # Periodic exploration and evaluation
        if episode % 10 == 0:
            # Random episode to maintain exploration
            dqn_agent.run_episode(explore_randomly=True)
            
            # Evaluation after warmup (episode 100+)
            if episode >= 100:
                eval_reward, eval_boss_hp = run_evaluation(dqn_agent, num_frames)
                
                # Check for best reward model
                if eval_reward > best_eval_reward:
                    print('  → New best evaluation model (by reward)!')
                    best_eval_reward = eval_reward
                    dqn_agent.save_checkpoint('best', online_only=True)
                
                # Track boss HP for averaging
                recent_boss_hp.append(eval_boss_hp)
                if len(recent_boss_hp) > 10:
                    recent_boss_hp.pop(0)
                
                avg_hp = sum(recent_boss_hp) / len(recent_boss_hp)
                print(f'  Avg boss HP (last {len(recent_boss_hp)} evals): {avg_hp*100:.1f}%')
                
                # Save best HP model (requires at least 3 evaluations)
                if avg_hp < best_boss_hp and len(recent_boss_hp) >= 3:
                    print(f'  → New best HP model! ({avg_hp*100:.1f}% < {best_boss_hp*100:.1f}%)')
                    best_boss_hp = avg_hp
                    dqn_agent.save_checkpoint('besthp', online_only=True)
                
                # Log HP metrics
                dqn_agent.log_metrics({
                    'eval/boss_hp': eval_boss_hp,
                    'eval/avg_boss_hp': avg_hp
                }, episode)
        
        # Save latest model
        dqn_agent.save_checkpoint('latest', online_only=True)
        
        # Log training metrics
        dqn_agent.log_metrics({
            'train/reward': reward,
            'train/loss': loss,
            'train/total_steps': dqn_agent.env_steps
        }, episode)
        
        # Progress report
        print(f'  Steps: {dqn_agent.env_steps}, Updates: {dqn_agent.gradient_steps}, ε: {dqn_agent.eps:.4f}')
        print(f'  Reward: {reward:.3f}, Loss: {loss:.5f}, LR: {lr:.8f}')
        print(f'  Memory: {psutil.virtual_memory().percent}%')
    
    # Save final model
    dqn_agent.save_checkpoint('final', online_only=False)
    
    print('\n' + '=' * 60)
    print('Training Complete!')
    print('=' * 60)
    print(f'Best evaluation reward: {best_eval_reward:.2f}')
    print(f'Best average boss HP: {best_boss_hp*100:.1f}%')


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Initialize components and start training."""
    print('=' * 60)
    print('Silksong RL - Lace Boss Training')
    print('=' * 60)
    
    # Configuration
    NUM_FRAMES = 4
    
    # Create environment
    env = environment.LaceEnvV2(
        obs_shape=(192, 192),
        rgb=False,
        gap=0.17,
        # Reward parameters
        damage_penalty=0.35,
        hit_reward=0.85,
        heal_reward=0.7,
        inactivity_penalty=0.3,
        inactivity_window=5.0,
        victory_base=18.0,
        defeat_base=10.0,
    )
    
    # Report action space
    print(f"\nAction space: {env.action_space.n} discrete actions")
    print(f"  Movement: {len(environment.Move)} options")
    print(f"  Attack: {len(environment.Attack)} options (attack/spell/tool)")
    print(f"  Displacement: {len(environment.Displacement)} options (jump/dash)")
    print(f"  Heal: {len(environment.Heal)} options")
    print()
    
    # Create Q-network
    network = create_network(env, NUM_FRAMES)
    
    # Create replay buffer with N-step returns
    replay_mem = replay.NStepReplayMemory(
        capacity=180000,
        n_steps=10,
        discount=0.99,
        priority_config=None
    )
    
    # Create DQN agent
    dqn_agent = agent.DQNAgent(
        env=env,
        replay_buffer=replay_mem,
        frame_stack=NUM_FRAMES,
        discount=0.99,
        epsilon=0.0,  # Using noisy networks instead
        epsilon_schedule=lambda eps, step: 0.0,
        target_sync_freq=8000,
        update_freq=4,
        network=network,
        learning_rate=8e-5,
        decay_lr=False,
        loss_fn=torch.nn.MSELoss(),
        batch_size=32,
        device=DEVICE,
        double_q=True,
        augment_data=True,
        use_svea=True,
        reset_interval=0,
        num_targets=1,
        experiment_name='Lace',
        disable_saving=False
    )
    
    # Start training
    train_agent(dqn_agent, NUM_FRAMES)


if __name__ == '__main__':
    main()

