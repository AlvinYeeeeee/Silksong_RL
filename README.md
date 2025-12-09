# Silksong RL - Lace Boss Agent

A reinforcement learning agent trained to defeat the Lace boss in *Hollow Knight: Silksong* using Deep Q-Networks (DQN).

## Overview

This project trains an AI to play the Lace boss fight by:
- Capturing game frames directly from the screen
- Processing 192×192 grayscale observations
- Making decisions at ~6 FPS using a CNN + Dueling DQN architecture
- Learning through iterative reward engineering

## Results

- **Win Rate**: 40% (4/10 episodes)
- **Average Boss Damage**: 79.2%

## Architecture

### Neural Network
- **Feature Extractor**: 5-layer CNN (4→32→48→96→160→320 channels)
- **Decision Head**: Dueling DQN with noisy linear layers
- **Output**: 11,520-dim features → 96 discrete actions

### DQN Improvements
| Technique | Benefit |
|-----------|---------|
| Double DQN | Reduces Q-value overestimation |
| Dueling DQN | Better value estimation |
| Noisy Networks | Built-in exploration |
| 10-step Returns | Faster credit assignment |
| DrQ (Data Aug) | Better generalization |
| Mixed Precision | 2× faster training |

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Silksong_RL.git
cd Silksong_RL

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
python run_training.py
```

Make sure:
1. Silksong is running in 1280×720 windowed mode
2. You are in the Lace boss fight arena
3. The game window is focused and visible

### Evaluation / Demo
```bash
python demo.py              # Run 10 episodes with best model
python demo.py best 20      # Run 20 episodes
python demo.py besthp       # Use the model that deals most damage
```

## Project Structure

```
Silksong_RL/
├── agent.py           # DQN training agent
├── demo.py            # Evaluation script
├── environment.py     # Gym environment for Silksong
├── networks.py        # CNN + Q-network architectures
├── priority_tree.py   # Sum tree for prioritized replay
├── replay.py          # Experience replay buffers
├── run_training.py    # Main training script
├── requirements.txt   # Python dependencies
└── CHANGELOG.md       # Development history
```

## Action Space

| Category | Options |
|----------|---------|
| Movement | No-op, Left, Right |
| Attack | No-op, Attack, Spell, Tool |
| Displacement | No-op, Short Jump, Long Jump, Dash |
| Heal | No-op, Heal |

Total: 3 × 4 × 4 × 2 = **96 discrete actions**

## Reward Engineering

The reward function was iteratively refined to address behavioral issues:

1. **Hiding Problem**: Agent hid in corner → Added heavy defeat penalties + inactivity punishment
2. **Tool Spam**: Agent used all tools immediately → Added 5-second cooldown
3. **Final Tuning**: Added survival rewards, heal bonuses, damage scaling

See [CHANGELOG.md](CHANGELOG.md) for detailed reward evolution.

## Requirements

- Python 3.10+
- CUDA-capable GPU (for training)
- Windows (for DirectInput game control)
- Hollow Knight: Silksong

## License

MIT License

## Acknowledgments

- Inspired by various video game RL projects
- Built with PyTorch, OpenAI Gym, and Kornia

