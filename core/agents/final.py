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
    pass

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
    pass


