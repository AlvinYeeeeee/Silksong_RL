"""
Evaluation script for testing trained Lace boss agents.

Use this script to:
- Test how well your trained agent performs
- Record gameplay videos for presentation
- Watch the agent play in real-time

Usage:
    python demo.py                           # Load 'best' from latest folder
    python demo.py besttrain                 # Load 'besttrain' model
    python demo.py latest                    # Load 'latest' model  
    python demo.py best 50                   # Run 50 episodes
    python demo.py 1764494530Lace best 20   # Specific folder, 20 episodes
    python demo.py path/to/model.pt          # Direct path to model file

Prerequisites:
    1. Silksong running in 1280x720 windowed mode
    2. Player in the Lace boss fight arena
    3. Game window focused and visible

Tips for Recording:
    - Start OBS/screen recorder BEFORE running this script
    - The agent plays continuously until all episodes complete
"""

import sys
import os
import time
import numpy as np
import torch
from collections import deque
from torch.backends import cudnn

import environment
import networks


# =============================================================================
# Configuration
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    cudnn.benchmark = True
print(f"Evaluation device: {DEVICE}")

# Must match training configuration
FRAME_STACK = 4
OBSERVATION_SIZE = (192, 192)


# =============================================================================
# Model Setup
# =============================================================================

def build_network(env, num_frames):
    """
    Construct Q-network matching training architecture.
    
    Args:
        env: Environment for observation/action space info
        num_frames: Number of stacked frames
        
    Returns:
        Q-network on evaluation device
    """
    channels, *spatial = env.observation_space.shape
    
    encoder = networks.ConvEncoder(
        spatial_shape=spatial,
        input_channels=num_frames * channels,
        activation='relu',
        use_spectral_norm=False
    )
    
    q_net = networks.DuelingQNetwork(
        encoder=encoder,
        num_actions=env.action_space.n,
        activation='relu',
        noisy=True,
        use_spectral_norm=False
    )
    
    return q_net.to(DEVICE)


def load_checkpoint(network, checkpoint_path):
    """
    Load model weights from checkpoint file.
    
    Args:
        network: Q-network to load weights into
        checkpoint_path: Path to .pt checkpoint file
        
    Returns:
        Network with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    weights = torch.load(checkpoint_path, map_location=DEVICE)
    network.load_state_dict(weights)
    network.eval()
    print("Checkpoint loaded successfully!")
    
    return network


# =============================================================================
# Evaluation Logic
# =============================================================================

def play_episode(env, network, verbose=True):
    """
    Run one episode with greedy action selection.
    
    Args:
        env: Game environment
        network: Trained Q-network
        verbose: Whether to print action details
        
    Returns:
        Tuple of (episode_reward, step_count, final_info)
    """
    # Disable exploration noise for deterministic evaluation
    network.set_exploration(False)
    
    obs, _ = env.reset()
    frames = deque([obs] * FRAME_STACK, maxlen=FRAME_STACK)
    
    total_reward = 0.0
    steps = 0
    
    while True:
        # Stack frames and normalize
        stacked = np.concatenate(tuple(frames), axis=0, dtype=np.float32)
        
        with torch.no_grad():
            obs_tensor = torch.from_numpy(stacked).unsqueeze(0).to(DEVICE)
            # CRITICAL: Match training normalization [-1, 1]
            obs_tensor = obs_tensor / 127.5 - 1.0
            q_values = network(obs_tensor)
            action = int(q_values.argmax(dim=-1).item())
        
        obs_next, reward, done, _, info = env.step(action)
        
        total_reward += reward
        steps += 1
        frames.append(obs_next)
        
        if done:
            break
    
    # Re-enable exploration for future use
    network.set_exploration(True)
    
    return total_reward, steps, info


def find_latest_experiment(base_path='./saved/'):
    """
    Locate the most recent experiment folder with model files.
    
    Args:
        base_path: Directory containing experiment folders
        
    Returns:
        Path to latest experiment folder, or None if not found
    """
    if not os.path.exists(base_path):
        return None
    
    # Look for Lace experiment folders
    experiments = [
        f for f in os.listdir(base_path) 
        if f.endswith('Lace') and os.path.isdir(os.path.join(base_path, f))
    ]
    
    if not experiments:
        return None
    
    # Sort by timestamp (folder name format: {timestamp}Lace)
    experiments.sort(reverse=True)
    
    # Find first folder with actual model files
    for exp in experiments:
        exp_path = os.path.join(base_path, exp)
        model_files = [f for f in os.listdir(exp_path) if f.endswith('.pt')]
        if model_files:
            return exp_path
    
    return None


# =============================================================================
# Main Execution
# =============================================================================

def main():
    # Default settings
    model_type = 'best'
    num_episodes = 10
    experiment_dir = None
    
    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg.endswith('.pt'):
            model_type = arg  # Direct path to checkpoint
        elif arg.isdigit():
            num_episodes = int(arg)
        elif os.path.isdir(arg) or os.path.isdir(f'./saved/{arg}'):
            experiment_dir = arg if os.path.isdir(arg) else f'./saved/{arg}'
        else:
            model_type = arg  # Model prefix (best, besttrain, etc.)
    
    # Resolve checkpoint path
    if model_type.endswith('.pt'):
        checkpoint = model_type
    else:
        if experiment_dir is None:
            experiment_dir = find_latest_experiment()
            if experiment_dir is None:
                print("ERROR: No experiments found in ./saved/")
                print("Train a model first with: python run_training.py")
                sys.exit(1)
        
        print(f"Experiment folder: {experiment_dir}")
        checkpoint = os.path.join(experiment_dir, f'{model_type}online.pt')
    
    # Initialize environment
    print("\nInitializing environment...")
    env = environment.LaceEnvV2(
        obs_shape=OBSERVATION_SIZE,
        rgb=False,
        gap=0.17,
    )
    
    # Build and load network
    print("Building network...")
    network = build_network(env, FRAME_STACK)
    network = load_checkpoint(network, checkpoint)
    
    # Countdown to start
    print("\n" + "=" * 60)
    print("🎮 AGENT EVALUATION MODE 🎮")
    print("=" * 60)
    print(f"\nPlaying {num_episodes} episode(s)")
    print("\n⚠️  Switch to the game window now!")
    print("   Make sure you are IN THE BOSS FIGHT")
    print("\n💡 Start your screen recorder now if making a video!")
    
    for i in range(5, 0, -1):
        print(f"   Starting in {i}...")
        time.sleep(1)
    print("   GO!\n")
    
    # Run evaluation episodes
    rewards = []
    victories = 0
    
    for ep in range(1, num_episodes + 1):
        print(f"\n{'='*40}")
        print(f"  EPISODE {ep}/{num_episodes}")
        print(f"{'='*40}")
        
        try:
            reward, steps, info = play_episode(env, network)
            rewards.append(reward)
            
            # Determine outcome (victory rewards are 10+)
            is_victory = reward > 5
            if is_victory:
                victories += 1
                print(f"🏆 VICTORY! Reward = {reward:.2f}, Steps = {steps}")
            else:
                print(f"💀 Defeat. Reward = {reward:.2f}, Steps = {steps}")
            
            hp = info.get('hornet_hp', '?')
            boss_hp = info.get('enemy_hp', 0) * 100
            print(f"   Hornet HP: {hp}/9, Boss HP: {boss_hp:.1f}%")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
            break
        except Exception as e:
            print(f"Episode error: {e}")
            continue
        
        if ep < num_episodes:
            time.sleep(1)  # Brief pause between episodes
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    if rewards:
        print(f"Episodes: {len(rewards)}")
        print(f"Win Rate: {victories}/{len(rewards)} ({100*victories/len(rewards):.1f}%)")
        print(f"Average Reward: {np.mean(rewards):.2f}")
        print(f"Best Episode: {np.max(rewards):.2f}")
        print(f"Worst Episode: {np.min(rewards):.2f}")
    
    env.close()
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

