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
    #elif torch.backends.mps.is_available():
        #print("Using MPS (Apple Silicon)")
        #return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

class DDQNNetwork(nn.Module):
    """Double DQN Network for MCTS-DDQN integration."""
    
    def __init__(self, state_dim=54, action_dim=12, hidden_dim=512):
        super(DDQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

class MCTSNode:
    """MCTS Node for DDQN-guided search."""
    
    def __init__(self, state: np.ndarray, parent: Optional['MCTSNode'] = None, 
                 action: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action = action
        
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.q_mcts: Dict[int, float] = {}  # MCTS Q-values for actions
        self.is_expanded = False
        
    def get_value(self) -> float:
        """Get average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def get_q_value(self, action: int) -> float:
        """Get MCTS Q-value for a specific action."""
        return self.q_mcts.get(action, 0.0)
    
    def select_child(self, c_puct: float, prior_probs: np.ndarray) -> Tuple[int, 'MCTSNode']:
        """Select child using UCT with neural network priors."""
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        for action in range(len(prior_probs)):
            if action in self.children:
                child = self.children[action]
                # UCT formula with neural network priors
                q_value = self.get_q_value(action)
                u_value = (c_puct * prior_probs[action] * 
                          math.sqrt(self.visits) / (1 + child.visits))
                score = q_value + u_value
            else:
                # Unvisited action gets high priority
                score = c_puct * prior_probs[action] * math.sqrt(self.visits)
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = self.children.get(action)
        
        return best_action, best_child
    
    def expand(self, action: int, new_state: np.ndarray):
        """Expand this node by adding a child for the given action."""
        child = MCTSNode(state=new_state, parent=self, action=action)
        self.children[action] = child
        self.is_expanded = True
        return child
    
    def backup(self, action: int, value: float):
        """Backup the value up the tree, updating MCTS Q-values."""
        self.visits += 1
        self.value_sum += value
        
        # Update MCTS Q-value for the action (if this is not the root)
        if action is not None:
            if action in self.q_mcts:
                # Get the child node for this action
                child = self.children.get(action)
                if child:
                    # Running average update
                    old_q = self.q_mcts[action]
                    n_visits = child.visits
                    self.q_mcts[action] = old_q + (value - old_q) / max(1, n_visits)
                else:
                    self.q_mcts[action] = value
            else:
                self.q_mcts[action] = value
        
        if self.parent and self.action is not None:
            self.parent.backup(self.action, value)

class MCTS_DDQN:
    """MCTS search guided by DDQN network."""
    
    def __init__(self, online_net: DDQNNetwork, env: RubiksCubeEnv, 
                 c_puct: float = 1.0, temperature_prior: float = 1.0,
                 device: torch.device = torch.device('cpu')):
        self.online_net = online_net
        self.env = env
        self.c_puct = c_puct
        self.temperature_prior = temperature_prior
        self.device = device
        
    def get_prior_probs(self, state: np.ndarray) -> np.ndarray:
        """Get prior probabilities from DDQN Q-values using softmax."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            # Convert Q-values to probabilities using softmax
            probs = F.softmax(q_values / self.temperature_prior, dim=1)
            return probs.cpu().numpy()[0]
    
    def get_state_value(self, state: np.ndarray) -> float:
        """Get state value estimate from DDQN (max Q-value)."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            return torch.max(q_values).item()
    
    def search(self, initial_state: np.ndarray, num_simulations: int = 200) -> Tuple[np.ndarray, Dict[int, float]]:
        """Run MCTS search and return improved policy and action Q-values."""
        root = MCTSNode(initial_state)
        
        # Create a temporary environment for MCTS simulations
        temp_env = RubiksCubeEnv(scrambles=1, max_steps=50)
        
        for _ in range(num_simulations):
            node = root
            path = [(node, None)]
            
            # Selection: traverse down the tree
            while node.is_expanded and node.children:
                prior_probs = self.get_prior_probs(node.state)
                action, child = node.select_child(self.c_puct, prior_probs)
                
                if child is None:
                    # Need to expand this action
                    temp_env.cube_state = node.state.reshape(6, 9)
                    temp_env._apply_action(action)
                    new_state = temp_env.cube_state.flatten()
                    child = node.expand(action, new_state)
        
            node = child
            path.append((node, action))
    
        # Expansion (if not already expanded)
        if not node.is_expanded:
            temp_env.cube_state = node.state.reshape(6, 9)
            if not temp_env._is_solved():
                prior_probs = self.get_prior_probs(node.state)
                action = np.random.choice(12, p=prior_probs)
                
                temp_env._apply_action(action)
                new_state = temp_env.cube_state.flatten()
                
                child = node.expand(action, new_state)
                node = child
                path.append((node, action))
        
        # Evaluation
        temp_env.cube_state = node.state.reshape(6, 9)
        if temp_env._is_solved():
            value = 1.0
        else:
            value = self.get_state_value(node.state)
        
        # Backup
        for i in range(len(path) - 1, -1, -1):
            node, action = path[i]
            if action is not None:
                node.backup(action, value)
            else:
                # Root node
                node.visits += 1
                node.value_sum += value

        # Generate improved policy from visit counts (MOVED INSIDE THE FUNCTION)
        visit_counts = np.zeros(12)
        q_values = {}

        for action in range(12):
            if action in root.children:
                visit_counts[action] = root.children[action].visits
                q_values[action] = root.get_q_value(action)
            else:
                visit_counts[action] = 0
                q_values[action] = 0.0

        # Convert visit counts to policy
        if visit_counts.sum() == 0:
            policy = np.ones(12) / 12
        else:
            policy = visit_counts / visit_counts.sum()

        return policy, q_values

class ReplayBuffer:
    """Replay buffer for MCTS-DDQN training."""
    
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)
    
    def store(self, state, action, reward, next_state, done, mcts_policy, mcts_q_values):
        """Store transition with MCTS policy information."""
        self.buffer.append((state, action, reward, next_state, done, mcts_policy, mcts_q_values))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

def generate_episode(mcts_ddqn: MCTS_DDQN, env: RubiksCubeEnv, 
                    temperature: float = 1.0, max_steps: int = 50) -> List[Tuple]:
    """Generate one episode using MCTS-guided policy."""
    episode_data = []
    state, _ = env.reset()  # Use proper gym interface
    step = 0
    
    while step < max_steps:
        current_state = state.copy()
        
        # Run MCTS search
        mcts_policy, mcts_q_values = mcts_ddqn.search(current_state, num_simulations=200)
        
        # Apply temperature to policy
        if temperature != 1.0:
            mcts_policy = mcts_policy ** (1.0 / temperature)
            mcts_policy = mcts_policy / mcts_policy.sum()
        
        # Sample action from MCTS policy
        action = np.random.choice(12, p=mcts_policy)
        
        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Enhanced reward shaping
        if terminated and reward > 0:  # Solved
            reward = 10.0  # Large positive reward for solving
        elif done and reward <= 0:  # Failed to solve
            reward = -1.0  # Penalty for failing
        else:
            reward = -0.01  # Small step penalty
        
        # Store transition
        episode_data.append((
            current_state,
            action,
            reward,
            next_state.copy(),
            done,
            mcts_policy.copy(),
            mcts_q_values.copy()
        ))
        
        state = next_state
        step += 1
        
        if done:
            break
    
    return episode_data

def train_ddqn(online_net: DDQNNetwork, target_net: DDQNNetwork, 
               replay_buffer: ReplayBuffer, optimizer: torch.optim.Optimizer,
               device: torch.device, batch_size: int = 64, gamma: float = 0.99) -> float:
    """Train DDQN with MCTS-improved targets."""
    if replay_buffer.size() < batch_size:
        return 0.0
    
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones, mcts_policies, mcts_q_values = zip(*batch)
    
    # Convert to tensors
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    
    # Current Q-values
    current_q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    
    # Double DQN target calculation
    with torch.no_grad():
        # Select actions using online network
        next_actions = torch.argmax(online_net(next_states), dim=1)
        # Evaluate using target network
        next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        targets = rewards + gamma * next_q_values * (1 - dones)
    
    # Calculate loss
    loss = F.mse_loss(current_q_values, targets)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

def main():
    """Main training loop for MCTS-DDQN."""
    device = get_device()
    
    # Hyperparameters
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
    BUFFER_SIZE = 50000
    GAMMA = 0.99
    C_PUCT = 1.0
    TEMPERATURE_PRIOR = 1.0
    TARGET_UPDATE_FREQ = 1000
    
    # Training parameters
    TRAINING_ITERATIONS = 100
    EPISODES_PER_ITERATION = 10
    TRAIN_STEPS_PER_ITERATION = 50
    
    # Curriculum learning
    INITIAL_SCRAMBLES = 1
    MAX_SCRAMBLES = 10
    SCRAMBLE_INCREMENT_FREQ = 10
    
    MODEL_PATH = "./models/mcts_ddqn_final.pth"
    
    # Initialize environment and networks
    env = RubiksCubeEnv(scrambles=INITIAL_SCRAMBLES, max_steps=50)
    online_net = DDQNNetwork().to(device)
    target_net = DDQNNetwork().to(device)
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    mcts_ddqn = MCTS_DDQN(online_net, env, C_PUCT, TEMPERATURE_PRIOR, device)
    
    print(f"Starting MCTS-DDQN training on {device}")
    print(f"Training iterations: {TRAINING_ITERATIONS}")
    
    episode_rewards = []
    episode_steps = []
    solved_episodes = 0
    total_episodes = 0
    
    for iteration in range(TRAINING_ITERATIONS):
        print(f"\n--- Training Iteration {iteration + 1}/{TRAINING_ITERATIONS} ---")
        
        # Curriculum learning: increase scrambles
        current_scrambles = min(INITIAL_SCRAMBLES + (iteration // SCRAMBLE_INCREMENT_FREQ), MAX_SCRAMBLES)
        env.scrambles = current_scrambles
        print(f"Current scramble depth: {current_scrambles}")
        
        # Generate episodes
        iteration_rewards = []
        iteration_steps = []
        iteration_solved = 0
        
        for episode in range(EPISODES_PER_ITERATION):
            # Temperature decay
            temperature = max(0.1, 1.0 - (iteration / TRAINING_ITERATIONS))
            
            episode_data = generate_episode(mcts_ddqn, env, temperature=temperature)
            
            # Store in replay buffer
            for transition in episode_data:
                replay_buffer.store(*transition)
            
            # Track statistics
            episode_reward = sum(transition[2] for transition in episode_data)
            episode_step_count = len(episode_data)
            
            iteration_rewards.append(episode_reward)
            iteration_steps.append(episode_step_count)
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step_count)
            
            if episode_reward > 0:  # Solved
                iteration_solved += 1
                solved_episodes += 1
            
            total_episodes += 1
        
        # Train the network
        total_loss = 0
        for _ in range(TRAIN_STEPS_PER_ITERATION):
            loss = train_ddqn(online_net, target_net, replay_buffer, optimizer, device, BATCH_SIZE, GAMMA)
            total_loss += loss
        
        avg_loss = total_loss / TRAIN_STEPS_PER_ITERATION if TRAIN_STEPS_PER_ITERATION > 0 else 0
        
        # Update target network
        if (iteration + 1) % (TARGET_UPDATE_FREQ // TRAIN_STEPS_PER_ITERATION) == 0:
            target_net.load_state_dict(online_net.state_dict())
            print("Target network updated")
        
        # Print statistics
        avg_reward = np.mean(iteration_rewards)
        avg_steps = np.mean(iteration_steps)
        solve_rate = (iteration_solved / EPISODES_PER_ITERATION) * 100
        overall_solve_rate = (solved_episodes / total_episodes) * 100
        
        print(f"Average reward: {avg_reward:.3f}")
        print(f"Average steps: {avg_steps:.1f}")
        print(f"Solve rate this iteration: {solve_rate:.1f}%")
        print(f"Overall solve rate: {overall_solve_rate:.1f}%")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Replay buffer size: {replay_buffer.size()}")
        
        # Save model periodically
        if (iteration + 1) % 20 == 0:
            torch.save(online_net.state_dict(), f"./models/mcts_ddqn_iter_{iteration+1}.pth")
            print(f"Model saved at iteration {iteration + 1}")
    
    # Save final model
    torch.save(online_net.state_dict(), MODEL_PATH)
    print(f"Training complete! Final model saved as {MODEL_PATH}")
    
    # Test the trained agent
    print("\nTesting the trained agent...")
    test_results = test_agent(mcts_ddqn, env, device, num_tests=20)

    # Save results with proper test data
    hyperparams = {
        "SCRAMBLES": MAX_SCRAMBLES,  # Add missing required fields
        "MAX_STEPS": 50,
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "BUFFER_SIZE": BUFFER_SIZE,
        "GAMMA": GAMMA,
        "LR": LEARNING_RATE,  # Alias for compatibility
        "C_PUCT": C_PUCT,
        "TEMPERATURE_PRIOR": TEMPERATURE_PRIOR,
        "TRAINING_ITERATIONS": TRAINING_ITERATIONS,
        "EPISODES_PER_ITERATION": EPISODES_PER_ITERATION,
        "INITIAL_SCRAMBLES": INITIAL_SCRAMBLES,
        "MAX_SCRAMBLES": MAX_SCRAMBLES,
        "NUM_EPISODES": TRAINING_ITERATIONS * EPISODES_PER_ITERATION,
        "DEVICE": str(device),
    }

    save_results_to_csv(
        train_rewards=episode_rewards,
        train_steps=episode_steps,
        test_rewards=test_results.get('test_rewards', []),
        test_steps=test_results.get('test_steps', []),
        test_solved=test_results.get('solved', 0),
        hyperparams=hyperparams,
        output_path="./results/mcts_ddqn_results.csv"  # Use output_path not filename
    )

def test_agent(mcts_ddqn: MCTS_DDQN, env: RubiksCubeEnv, device: torch.device, num_tests: int = 10):
    """Test the trained MCTS-DDQN agent."""
    mcts_ddqn.online_net.eval()
    
    solved = 0
    total_moves = 0
    test_rewards = []
    test_steps = []
    test_scrambles = [3, 5, 7, 10]
    
    for scrambles in test_scrambles:
        env.scrambles = scrambles
        scramble_solved = 0
        
        print(f"\nTesting on {scrambles}-move scrambles:")
        
        tests_per_scramble = max(1, num_tests // len(test_scrambles))
        for test in range(tests_per_scramble):
            env.reset()
            moves = 0
            max_moves = 30
            episode_reward = 0
            
            for move in range(max_moves):
                current_state = env.cube_state.flatten()
                
                # Use MCTS with greedy action selection
                mcts_policy, _ = mcts_ddqn.search(current_state, num_simulations=100)
                action = np.argmax(mcts_policy)
                
                env._apply_action(action)
                moves += 1
                
                if env._is_solved():
                    solved += 1
                    scramble_solved += 1
                    total_moves += moves
                    episode_reward = 10.0  # Positive reward for solving
                    print(f"  Test {test + 1}: Solved in {moves} moves")
                    break
            else:
                total_moves += max_moves
                episode_reward = -1.0  # Negative reward for not solving
                print(f"  Test {test + 1}: Not solved within {max_moves} moves")
            
            test_rewards.append(episode_reward)
            test_steps.append(moves)
        
        print(f"Solved {scramble_solved}/{tests_per_scramble} cubes with {scrambles} scrambles")
    
    avg_moves = total_moves / len(test_rewards)
    overall_solve_rate = (solved / len(test_rewards)) * 100
    print(f"\nOverall Results: {solved}/{len(test_rewards)} solved ({overall_solve_rate:.1f}%), Average moves: {avg_moves:.1f}")
    
    mcts_ddqn.online_net.train()
    
    return {
        'solved': solved,
        'test_rewards': test_rewards,
        'test_steps': test_steps,
        'solve_rate': overall_solve_rate
    }

if __name__ == "__main__":
    main()