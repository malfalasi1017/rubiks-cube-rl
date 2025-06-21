import math
import random
import time
from collections import deque
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gymnasium_env.env.rubiks_cube import RubiksCubeEnv
from helpers.record_results import save_results_to_csv

def get_device():
    """Automatically detect and return the best available device"""
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

class AlphaZeroNetwork(nn.Module):
    """Neural network for AlphaZero-style MCTS."""
    
    def __init__(self, state_dim=54, action_dim=12, hidden_dim=512):
        super(AlphaZeroNetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head (state value)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1),
            nn.Tanh()
        )
    
    def forward(self, state):
        shared_features = self.shared(state)
        policy = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return policy, value

class MCTSNodeNN:
    """MCTS Node enhanced with neural network guidance."""
    
    def __init__(self, state: np.ndarray, parent: Optional['MCTSNodeNN'] = None, 
                 action: Optional[int] = None, prior_prob: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        
        self.children: Dict[int, 'MCTSNodeNN'] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def get_value(self) -> float:
        """Get average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNodeNN':
        """Select child using PUCT (Polynomial Upper Confidence Trees) formula."""
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            q_value = child.get_value()
            u_value = (c_puct * child.prior_prob * 
                      math.sqrt(self.visits) / (1 + child.visits))
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, action_probs: np.ndarray, env: RubiksCubeEnv):
        """Expand this node by adding all possible children."""
        self.is_expanded = True
        
        for action in range(len(action_probs)):
            # Create a copy of the environment to simulate the action
            env.cube_state = self.state.reshape(6, 9)
            env._apply_action(action)
            new_state = env.cube_state.flatten()
            
            # Add child with prior probability from neural network
            child = MCTSNodeNN(
                state=new_state,
                parent=self,
                action=action,
                prior_prob=action_probs[action]
            )
            self.children[action] = child
    
    def backup(self, value: float):
        """Backup the value up the tree."""
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(value)

class AlphaZeroMCTS:
    """AlphaZero-style MCTS with neural network guidance."""
    
    def __init__(self, network: AlphaZeroNetwork, env: RubiksCubeEnv, 
                 c_puct: float = 1.0, device: torch.device = torch.device('cpu')):
        self.network = network
        self.env = env
        self.c_puct = c_puct
        self.device = device
        
    def search(self, initial_state: np.ndarray, num_simulations: int = 800) -> Tuple[np.ndarray, float]:
        """Run MCTS search and return action probabilities and root value."""
        root = MCTSNodeNN(initial_state)
        
        for _ in range(num_simulations):
            node = root
            path = [node]
            
            # Selection: traverse down the tree
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                path.append(node)
            
            # Expansion and Evaluation
            self.env.cube_state = node.state.reshape(6, 9)
            
            # Check if terminal
            if self.env._is_solved():
                value = 1.0
            else:
                # Get neural network predictions
                state_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_probs, value_tensor = self.network(state_tensor)
                    action_probs = action_probs.cpu().numpy()[0]
                    value = value_tensor.cpu().item()
                
                # Expand the node
                if not node.is_expanded:
                    node.expand(action_probs, self.env)
            
            # Backup
            for node in path:
                node.backup(value)
        
        # Return action probabilities based on visit counts
        if not root.children:
            return np.ones(12) / 12, root.get_value()
        
        visits = np.array([root.children.get(a, MCTSNodeNN(np.array([]))).visits 
                          for a in range(12)])
        
        # Temperature-based action selection (temperature = 1 for training)
        if visits.sum() == 0:
            action_probs = np.ones(12) / 12
        else:
            action_probs = visits / visits.sum()
        
        return action_probs, root.get_value()

def generate_training_data(network: AlphaZeroNetwork, env: RubiksCubeEnv, 
                          num_games: int = 100, device: torch.device = torch.device('cpu')):
    """Generate training data through self-play."""
    training_data = []
    mcts = AlphaZeroMCTS(network, env, device=device)
    
    solved_games = 0
    
    for game in range(num_games):
        game_data = []
        env.reset()
        
        # Vary scramble difficulty during training
        scramble_depth = min(5 + (game // 10), 15)  # Gradually increase difficulty
        
        max_moves = 100  # Increased from 50
        for move in range(max_moves):
            current_state = env.cube_state.flatten()
            
            # Use temperature for exploration early in training
            temperature = max(0.1, 1.0 - (game / num_games))  # Decrease over time
            
            # Run MCTS to get action probabilities
            action_probs, _ = mcts.search(current_state, num_simulations=600)  # Increased
            
            # Apply temperature
            if temperature != 1.0:
                action_probs = action_probs ** (1.0 / temperature)
                action_probs = action_probs / action_probs.sum()
            
            # Store state and action probabilities
            game_data.append((current_state.copy(), action_probs.copy()))
            
            # Sample action based on probabilities
            action = np.random.choice(12, p=action_probs)
            
            # Apply action
            env._apply_action(action)
            
            # Check if solved
            if env._is_solved():
                solved_games += 1
                # Better reward assignment
                for i, (state, probs) in enumerate(game_data):
                    # Exponential decay based on distance from solution
                    reward = 1.0 * (0.9 ** (len(game_data) - i - 1))
                    training_data.append((state, probs, reward))
                print(f"Game {game + 1}: SOLVED in {len(game_data)} moves!")
                break
        else:
            # Partial rewards for good positions even if not solved
            for i, (state, probs) in enumerate(game_data):
                # Small negative reward, but less harsh
                reward = -0.01 * (i + 1) / len(game_data)  # Penalty increases with move count
                training_data.append((state, probs, reward))
        
        if (game + 1) % 10 == 0:
            solve_rate = (solved_games / (game + 1)) * 100
            print(f"Generated {game + 1}/{num_games} games - Solve rate: {solve_rate:.1f}%")
    
    print(f"Final solve rate: {(solved_games / num_games) * 100:.1f}%")
    return training_data

def train_network(network: AlphaZeroNetwork, training_data: List[Tuple], 
                 optimizer: torch.optim.Optimizer, device: torch.device, 
                 batch_size: int = 64, epochs: int = 5):
    """Train the neural network on the generated data."""
    network.train()
    
    # Convert training data to tensors
    states = torch.tensor([data[0] for data in training_data], dtype=torch.float32).to(device)
    target_probs = torch.tensor([data[1] for data in training_data], dtype=torch.float32).to(device)
    target_values = torch.tensor([data[2] for data in training_data], dtype=torch.float32).to(device)
    
    dataset = torch.utils.data.TensorDataset(states, target_probs, target_values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_policy_loss = 0
    total_value_loss = 0
    total_batches = 0
    
    for epoch in range(epochs):
        epoch_policy_loss = 0
        epoch_value_loss = 0
        
        for batch_states, batch_probs, batch_values in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            pred_probs, pred_values = network(batch_states)
            
            # Calculate losses separately
            policy_loss = F.kl_div(
                F.log_softmax(pred_probs, dim=1), 
                batch_probs, 
                reduction='batchmean'
            )
            value_loss = F.mse_loss(pred_values.squeeze(), batch_values)
            
            # Combined loss
            loss = policy_loss + value_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_batches += 1
        
        # Detailed logging
        avg_policy = epoch_policy_loss / len(dataloader)
        avg_value = epoch_value_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Policy Loss: {avg_policy:.4f}, Value Loss: {avg_value:.4f}")
    
    return (total_policy_loss + total_value_loss) / total_batches

def main():
    """Main training loop for AlphaZero-style learning."""
    device = get_device()
    
    # Hyperparameters - MODIFIED
    LEARNING_RATE = 0.0003  # Reduced from 0.001
    BATCH_SIZE = 64
    TRAINING_ITERATIONS = 50
    GAMES_PER_ITERATION = 25
    TRAINING_EPOCHS = 5
    MODEL_PATH = "./models/alphazero_cube.pth"
    
    # Initialize environment and network
    env = RubiksCubeEnv(scrambles=5, max_steps=50)
    network = AlphaZeroNetwork().to(device)
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"Starting AlphaZero training on {device}")
    print(f"Training iterations: {TRAINING_ITERATIONS}")
    print(f"Games per iteration: {GAMES_PER_ITERATION}")
    
    for iteration in range(TRAINING_ITERATIONS):
        print(f"\n--- Training Iteration {iteration + 1}/{TRAINING_ITERATIONS} ---")
        
        # Generate training data through self-play
        print("Generating training data...")
        training_data = generate_training_data(
            network, env, num_games=GAMES_PER_ITERATION, device=device
        )
        
        # Train the network
        print("Training network...")
        avg_loss = train_network(
            network, training_data, optimizer, device, 
            batch_size=BATCH_SIZE, epochs=TRAINING_EPOCHS
        )
        
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save model periodically
        if (iteration + 1) % 10 == 0:
            torch.save(network.state_dict(), f"{MODEL_PATH}_iter_{iteration + 1}")
            print(f"Model saved at iteration {iteration + 1}")
        
        # Test the network occasionally
        if (iteration + 1) % 5 == 0:
            print("Testing current model...")
            test_network(network, env, device, num_tests=5)
        
        # Step the scheduler after each iteration
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.6f}")
    
    # Save final model
    torch.save(network.state_dict(), MODEL_PATH)
    print(f"Training complete! Final model saved as {MODEL_PATH}")

def test_network(network: AlphaZeroNetwork, env: RubiksCubeEnv, 
                device: torch.device, num_tests: int = 10):
    """Test the trained network."""
    network.eval()
    mcts = AlphaZeroMCTS(network, env, device=device)
    
    solved = 0
    total_moves = 0
    
    for test in range(num_tests):
        env.reset()
        moves = 0
        max_moves = 30
        
        for move in range(max_moves):
            current_state = env.cube_state.flatten()
            
            # Use MCTS with lower simulation count for faster testing
            action_probs, _ = mcts.search(current_state, num_simulations=100)
            
            # Choose best action (greedy)
            action = np.argmax(action_probs)
            env._apply_action(action)
            moves += 1
            
            if env._is_solved():
                solved += 1
                total_moves += moves
                print(f"Test {test + 1}: Solved in {moves} moves")
                break
        else:
            total_moves += max_moves
            print(f"Test {test + 1}: Not solved within {max_moves} moves")
    
    avg_moves = total_moves / num_tests
    print(f"Results: {solved}/{num_tests} solved, Average moves: {avg_moves:.1f}")
    
    network.train()  # Switch back to training mode

if __name__ == "__main__":
    main()