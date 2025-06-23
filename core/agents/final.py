from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gymnasium_env.env.rubiks_cube import RubiksCubeEnv
from helpers.record_results import save_results_to_csv

def get_device():
    """Detect the best avaialable device for PyTorch"""
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(DQNNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),\
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class MCTSNode:
    pass

class MCTS_DDQN:
    pass

class ReplayBuffer:
    """Replay buffer for MCTS-DDQN training."""
    pass
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def store(self, state, action, reward, next_state, done, mcts_policy, mcts_q_values):
        """Store transition with MCTS policy information."""
        self.buffer.append((state, action, reward, next_state, done, mcts_policy, mcts_q_values))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

def main():
    device = get_device()

    # Hyperparameters
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
    BUFFER_SIZE = 50000
    GAMMA = 0.99
    C_PUCT = 1.0
    TEMPERATURE_PRIOR = 1.0
    TARGET_UPDATE_FREQ = 1000

    # Training Parameters
    TRAINING_ITERATIONS = 100
    EPISODES_PER_ITERATION = 10
    TRAIN_STEPS_PER_ITERATION = 50

    # Curriculum Learning
    INITIAL_SCRAMBLES = 1
    MAX_SCRAMBLES = 20

    MODEL_PATH = "./models/mcts_ddqn_final_modal.pth"

    # Initialze Environment and networks
    env = RubiksCubeEnv(scrambles=INITIAL_SCRAMBLES, max_steps=50)
    online_net = DQNNetwork().to(device)
    target_net = DQNNetwork().to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    episode_rewards = []
    episode_steps = []
    solved_episodes = 0
    total_episodes = 0

    for iteration in range(TRAINING_ITERATIONS):
        print(f"\n--- Training Iteration {iteration + 1}/{TRAINING_ITERATIONS} ---")

        # Curriculum learning: incresase scrambles

    torch.save(online_net.state_dict(), MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")



