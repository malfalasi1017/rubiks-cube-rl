from gymnasium_env.env.rubiks_cube import RubiksCubeEnv
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import random
import time

from helpers.record_results import save_results_to_csv

# Add device detection function
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

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        # Improved network architecture for better learning
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

class CurriculumManager:
    """Manages curriculum learning by adjusting scramble difficulty"""
    def __init__(self, min_scrambles=1, max_scrambles=20, initial_scrambles=1, 
                 success_threshold=0.8, min_episodes_per_level=500, 
                 evaluation_episodes=100):
        self.min_scrambles = min_scrambles
        self.max_scrambles = max_scrambles
        self.current_scrambles = initial_scrambles
        self.success_threshold = success_threshold
        self.min_episodes_per_level = min_episodes_per_level
        self.evaluation_episodes = evaluation_episodes
        
        # Tracking variables
        self.episodes_at_current_level = 0
        self.recent_success_rate = 0.0
        self.level_history = []
        self.success_history = []
        
    def should_evaluate(self):
        """Check if we should evaluate the current performance"""
        return self.episodes_at_current_level >= self.min_episodes_per_level
    
    def evaluate_performance(self, recent_rewards):
        """Evaluate performance on recent episodes"""
        if len(recent_rewards) < self.evaluation_episodes:
            return False, 0.0
            
        # Consider multiple success criteria for more nuanced evaluation
        recent_episodes = recent_rewards[-self.evaluation_episodes:]
        
        # Success rate 1: Solved cubes (reward > 0)
        solved_rate = np.mean([1 if r > 0 else 0 for r in recent_episodes])
        
        # Success rate 2: Average reward improvement (relative to random)
        avg_reward = np.mean(recent_episodes)
        
        # Success rate 3: Positive progress (reward > -1, meaning not worst case)
        progress_rate = np.mean([1 if r > -1 else 0 for r in recent_episodes])
        
        # Combined success metric - more lenient
        combined_success = max(solved_rate, progress_rate * 0.5, min(avg_reward / 10 + 0.5, 1.0))
        
        self.recent_success_rate = combined_success
        
        print(f"   📊 Evaluation: Solved={solved_rate:.3f}, Progress={progress_rate:.3f}, "
              f"Avg Reward={avg_reward:.3f}, Combined={combined_success:.3f}")
        
        return combined_success >= self.success_threshold, combined_success
    
    def update_curriculum(self, recent_rewards):
        """Update curriculum based on performance"""
        if not self.should_evaluate():
            return False, self.current_scrambles
            
        can_advance, success_rate = self.evaluate_performance(recent_rewards)
        
        if can_advance and self.current_scrambles < self.max_scrambles:
            old_scrambles = self.current_scrambles
            self.current_scrambles = min(self.current_scrambles + 1, self.max_scrambles)
            self.episodes_at_current_level = 0
            
            print(f"\n🎓 CURRICULUM UPDATE: {old_scrambles} → {self.current_scrambles} scrambles")
            print(f"   Success rate: {success_rate:.3f} (threshold: {self.success_threshold})")
            
            # Record history
            self.level_history.append((old_scrambles, success_rate))
            return True, self.current_scrambles
        else:
            if self.current_scrambles >= self.max_scrambles:
                print(f"\n✅ Maximum curriculum level reached ({self.max_scrambles} scrambles)")
            else:
                print(f"\n📊 Staying at {self.current_scrambles} scrambles (success rate: {success_rate:.3f})")
            return False, self.current_scrambles
    
    def increment_episode(self):
        """Increment episode counter for current level"""
        self.episodes_at_current_level += 1
    
    def get_current_scrambles(self):
        """Get current scramble difficulty"""
        return self.current_scrambles
    
    def get_progress_info(self):
        """Get curriculum progress information"""
        return {
            'current_scrambles': self.current_scrambles,
            'episodes_at_level': self.episodes_at_current_level,
            'recent_success_rate': self.recent_success_rate,
            'level_history': self.level_history,
            'progress': f"{self.current_scrambles}/{self.max_scrambles}"
        }

def select_action(state, online_net, epsilon, action_dim, device):
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = online_net(state_tensor)
            return torch.argmax(q_values).item()

def update_target(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())

def main():
    # Get device
    device = get_device()
    
    # Hyperparameters - OPTIMIZED FOR CURRICULUM LEARNING
    MAX_STEPS = 60  # Increased to give more time to solve
    BUFFER_SIZE = 50_000  # Smaller buffer for faster updates
    BATCH_SIZE = 64  # Smaller batch for more frequent updates
    GAMMA = 0.99  # Slightly lower discount for more immediate focus
    LR = 1e-3  # Higher learning rate for faster initial learning
    EPS_START = 0.9  # Start with more exploration
    EPS_END = 0.1  # Keep some exploration
    EPS_DECAY = 0.995  # Slower decay to maintain exploration longer
    TARGET_UPDATE_FREQ = 500  # More frequent target updates
    TRAIN_START = 500  # Start training sooner
    NUM_EPISODES = 100_000  # Increased for curriculum learning
    MODEL_PATH = "./models/dqn_cl_cube.pth"
    RESULTS_PATH = "./results/dqn_cl_results.csv"
    
    # Create directories if they don't exist
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # Curriculum Learning Parameters - ADJUSTED FOR EASIER PROGRESSION
    MIN_SCRAMBLES = 1
    MAX_SCRAMBLES = 20
    INITIAL_SCRAMBLES = 1
    SUCCESS_THRESHOLD = 0.8 
    MIN_EPISODES_PER_LEVEL = 1000  # More episodes to stabilize learning
    EVALUATION_EPISODES = 200  # More episodes for evaluation
    
    # Initialize curriculum manager
    curriculum = CurriculumManager(
        min_scrambles=MIN_SCRAMBLES,
        max_scrambles=MAX_SCRAMBLES,
        initial_scrambles=INITIAL_SCRAMBLES,
        success_threshold=SUCCESS_THRESHOLD,
        min_episodes_per_level=MIN_EPISODES_PER_LEVEL,
        evaluation_episodes=EVALUATION_EPISODES
    )
    
    # Initialize environment with starting scrambles
    env = RubiksCubeEnv(scrambles=curriculum.get_current_scrambles(), max_steps=MAX_STEPS)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Move networks to device
    online_net = DQNNetwork(state_dim, action_dim).to(device)
    target_net = DQNNetwork(state_dim, action_dim).to(device)
    update_target(online_net, target_net)
    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPS_START
    total_steps = 0
    episode_rewards = []
    episode_steps = []
    curriculum_changes = []

    print(f"🚀 Starting Curriculum Learning DQN Training")
    print(f"   Initial scrambles: {curriculum.get_current_scrambles()}")
    print(f"   Target scrambles: {MAX_SCRAMBLES}")
    print(f"   Success threshold: {SUCCESS_THRESHOLD}")
    print(f"   Episodes per level: {MIN_EPISODES_PER_LEVEL}")
    print(f"   Network architecture: {state_dim} -> 512 -> 512 -> 256 -> {action_dim}")
    print(f"   Device: {device}")

    # Warmup phase - collect initial experience
    print("\n🔥 Warmup phase: Collecting initial experience...")
    warmup_episodes = 100
    for warmup_ep in range(warmup_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.random.randint(action_dim)  # Random actions for diversity
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.store((state, action, reward, next_state, done))
            state = next_state
        if warmup_ep % 20 == 0:
            print(f"   Warmup episode {warmup_ep}/{warmup_episodes}, Buffer size: {replay_buffer.size()}")
    
    print(f"✅ Warmup complete! Buffer size: {replay_buffer.size()}")
    print("\n🎓 Starting curriculum learning...")

    for episode in range(1, NUM_EPISODES + 1):
        # Update curriculum if needed
        curriculum_updated, new_scrambles = curriculum.update_curriculum(episode_rewards)
        if curriculum_updated:
            # Update environment with new scramble difficulty
            env = RubiksCubeEnv(scrambles=new_scrambles, max_steps=MAX_STEPS)
            curriculum_changes.append({
                'episode': episode,
                'new_scrambles': new_scrambles,
                'success_rate': curriculum.recent_success_rate
            })
        
        curriculum.increment_episode()
        
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = select_action(state, online_net, epsilon, action_dim, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.store((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1
            total_steps += 1

            # OPTIMIZED: Train more frequently for better learning
            if replay_buffer.size() > TRAIN_START and total_steps % 2 == 0:  # Train every 2 steps
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Create tensors directly on device (more efficient)
                states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                # Double DQN target calculation
                q_values = online_net(states).gather(1, actions).squeeze()
                
                with torch.no_grad():  # More efficient target calculation
                    next_actions = torch.argmax(online_net(next_states), dim=1, keepdim=True)
                    next_q_values = target_net(next_states).gather(1, next_actions).squeeze()
                    targets = rewards + GAMMA * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=0.5)
                optimizer.step()

                # Update target network
                if total_steps % TARGET_UPDATE_FREQ == 0:
                    update_target(online_net, target_net)

        # Decay epsilon
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if episode % 50 == 0:  # More frequent progress reports
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            progress_info = curriculum.get_progress_info()
            recent_success_rate = np.mean([1 if r > 0 else 0 for r in episode_rewards[-100:]]) if len(episode_rewards) >= 100 else 0
            recent_progress_rate = np.mean([1 if r > -1 else 0 for r in episode_rewards[-100:]]) if len(episode_rewards) >= 100 else 0
            
            print(f"Episode {episode:5d} | Scrambles: {progress_info['current_scrambles']:2d} | "
                  f"Avg Reward: {avg_reward:6.3f} | Solved: {recent_success_rate:.3f} | "
                  f"Progress: {recent_progress_rate:.3f} | Epsilon: {epsilon:.3f} | "
                  f"Level Episodes: {progress_info['episodes_at_level']:4d}")
            
            # Early success detection - show when model starts improving
            if recent_progress_rate > 0.1:
                print(f"   🎯 Model showing improvement! Progress rate: {recent_progress_rate:.3f}")

    # Save the trained model (move to CPU first for compatibility)
    torch.save(online_net.cpu().state_dict(), MODEL_PATH)
    online_net.to(device)  # Move back to device for testing
    print(f"Training complete. Model saved as {MODEL_PATH}")

    # Test the trained agent at different scramble levels
    print("\nTesting the trained agent at different scramble levels...")
    test_scramble_levels = [1, 5, 10, 15, 20]
    test_results = {}
    
    for scrambles in test_scramble_levels:
        if scrambles > curriculum.get_current_scrambles():
            continue  # Skip levels not reached during training
            
        print(f"\nTesting with {scrambles} scrambles...")
        test_env = RubiksCubeEnv(scrambles=scrambles, max_steps=MAX_STEPS)
        test_episodes = 20
        solved = 0
        test_episode_rewards = []
        test_episode_steps = []
        
        for i in range(test_episodes):
            state, _ = test_env.reset()
            done = False
            total_reward = 0
            steps = 0
            while not done and steps < MAX_STEPS:
                action = select_action(state, online_net, 0.0, action_dim, device)  # Greedy
                next_state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward
                steps += 1
                
            if reward > 0:
                solved += 1
            test_episode_rewards.append(total_reward)
            test_episode_steps.append(steps)
            
        success_rate = solved / test_episodes
        avg_steps = np.mean(test_episode_steps)
        avg_reward = np.mean(test_episode_rewards)
        
        test_results[scrambles] = {
            'solved': solved,
            'total': test_episodes,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward
        }
        
        print(f"  Solved {solved}/{test_episodes} cubes ({success_rate:.1%})")
        print(f"  Average steps: {avg_steps:.1f}")
        print(f"  Average reward: {avg_reward:.2f}")

    # Print curriculum learning summary
    print("\n" + "="*60)
    print("CURRICULUM LEARNING SUMMARY")
    print("="*60)
    print(f"Final scramble level reached: {curriculum.get_current_scrambles()}/{MAX_SCRAMBLES}")
    print(f"Total curriculum changes: {len(curriculum_changes)}")
    
    if curriculum_changes:
        print("\nCurriculum progression:")
        for change in curriculum_changes:
            print(f"  Episode {change['episode']}: Advanced to {change['new_scrambles']} scrambles "
                  f"(success rate: {change['success_rate']:.3f})")
    
    print("\nTest results by scramble level:")
    for scrambles, results in test_results.items():
        print(f"  {scrambles:2d} scrambles: {results['success_rate']:.1%} success rate")

    # Save results to CSV (including curriculum information)
    hyperparams = {
        "MIN_SCRAMBLES": MIN_SCRAMBLES,
        "MAX_SCRAMBLES": MAX_SCRAMBLES,
        "INITIAL_SCRAMBLES": INITIAL_SCRAMBLES,
        "SUCCESS_THRESHOLD": SUCCESS_THRESHOLD,
        "MIN_EPISODES_PER_LEVEL": MIN_EPISODES_PER_LEVEL,
        "EVALUATION_EPISODES": EVALUATION_EPISODES,
        "FINAL_SCRAMBLES": curriculum.get_current_scrambles(),
        "CURRICULUM_CHANGES": len(curriculum_changes),
        "MAX_STEPS": MAX_STEPS,
        "BUFFER_SIZE": BUFFER_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "GAMMA": GAMMA,
        "LR": LR,
        "EPS_START": EPS_START,
        "EPS_END": EPS_END,
        "EPS_DECAY": EPS_DECAY,
        "TARGET_UPDATE_FREQ": TARGET_UPDATE_FREQ,
        "TRAIN_START": TRAIN_START,
        "NUM_EPISODES": NUM_EPISODES,
        "MODEL_PATH": MODEL_PATH,
        "DEVICE": str(device),
    }
    
    # Use the final test results (highest scramble level reached)
    final_scrambles = curriculum.get_current_scrambles()
    if final_scrambles in test_results:
        final_test = test_results[final_scrambles]
        save_results_to_csv(
            train_rewards=episode_rewards,
            train_steps=episode_steps,
            test_rewards=[final_test['avg_reward']] * final_test['total'],
            test_steps=[final_test['avg_steps']] * final_test['total'],
            test_solved=final_test['solved'],
            hyperparams=hyperparams,
            output_path=RESULTS_PATH
        )

if __name__ == "__main__":
    main()
