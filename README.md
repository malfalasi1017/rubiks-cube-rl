# Rubik's Cube Reinforcement Learning

A reinforcement learning environment for the Rubik's Cube puzzle using Gymnasium.

## Overview

This project implements a Gymnasium environment for training reinforcement learning agents to solve the Rubik's Cube puzzle. The environment provides a standard interface with observation space, action space, rewards, and 3D visualization.

## Features

- Custom Gymnasium environment for Rubik's Cube
- 12 possible actions (clockwise and counter-clockwise rotations for each face)
- 3D visualization using Open3D
- Configurable scrambling complexity

## Installation with UV

This guide uses [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver written in Rust that significantly speeds up dependency management.

### Step-by-Step Setup

1. **Install uv** (if you don't have it already):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository**:

```bash
git clone https://github.com/yourusername/rubiks-cube-rl.git
cd rubiks-cube-rl
```

3. **Create and activate a virtual environment with uv**:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. **Install dependencies with uv**:

```bash
uv pip install -e .
```

## Running the Project

1. **Run the main example**:

```bash
python main.py
```

2. **Or use the environment in your own code**:

```python
import gymnasium as gym
import gymnasium_env

# Create the environment
env = gym.make("gymnasium_env/RubiksCube-v0", scrambles=5, render_mode="human")

# Reset the environment
observation, info = env.reset()

# Run an episode
done = False
while not done:
    # Take a random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Render the current state
    env.render()

env.close()
```

## Environment Details

- **Observation Space**: 54 discrete values (6 faces × 9 squares) representing the color of each square
- **Action Space**: 12 discrete actions representing face rotations
  - FRONT_CW, FRONT_CCW
  - RIGHT_CW, RIGHT_CCW
  - UP_CW, UP_CCW
  - LEFT_CW, LEFT_CCW
  - DOWN_CW, DOWN_CCW
  - BACK_CW, BACK_CCW
- **Reward**: Reward increases as more squares are in the correct position

## Project Structure

```
rubiks-cube-rl/
├── gymnasium_env/         # The custom Gymnasium environment
│   ├── __init__.py        # Environment registration
│   └── env/               # Environment implementation
│       ├── __init__.py
│       └── rubiks_cube.py # Main environment class
├── main.py                # Entry point script
├── pyproject.toml         # Project metadata and dependencies
└── requirements.txt       # Full list of dependencies
```

## Dependencies

- Python 3.12+
- gymnasium 1.1.1
- numpy 2.2.4
- open3d 0.19.0
- pygame 2.6.1

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Similar code found with 2 license types
