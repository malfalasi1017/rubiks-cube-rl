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

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

def select_action(state, online_net, epsilon, action_dim):
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = online_net(state_tensor)
            return torch.argmax(q_values).item()

def update_target(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())

def main():
    # Hyperparameters
    SCRAMBLES = 2
    MAX_STEPS = 3
    BUFFER_SIZE = 100_000
    BATCH_SIZE = 256
    GAMMA = 0.995
    LR = 1e-3
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.9997
    TARGET_UPDATE_FREQ = 1000
    TRAIN_START = 500
    NUM_EPISODES = 10_000
    MODEL_PATH = "./models/dqn_cube.pth"
    RESULTS_PATH = "./results/dqn_results.csv"

    env = RubiksCubeEnv(scrambles=SCRAMBLES, max_steps=MAX_STEPS)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    online_net = DQNNetwork(state_dim, action_dim)
    target_net = DQNNetwork(state_dim, action_dim)
    update_target(online_net, target_net)
    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPS_START
    total_steps = 0
    episode_rewards = []
    episode_steps = []

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = select_action(state, online_net, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.store((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1
            total_steps += 1

            # Training step
            if replay_buffer.size() > TRAIN_START:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Double DQN target calculation
                q_values = online_net(states).gather(1, actions).squeeze()
                next_actions = torch.argmax(online_net(next_states), dim=1, keepdim=True)
                next_q_values = target_net(next_states).gather(1, next_actions).squeeze()
                targets = rewards + GAMMA * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target network
                if total_steps % TARGET_UPDATE_FREQ == 0:
                    update_target(online_net, target_net)

        # Decay epsilon
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode} | Avg Reward (last 100): {avg_reward:.3f} | Epsilon: {epsilon:.3f}")

    # Save the trained model
    torch.save(online_net.state_dict(), MODEL_PATH)
    print(f"Training complete. Model saved as {MODEL_PATH}")

    # Test the trained agent
    print("\nTesting the trained agent...")
    test_episodes = 10
    solved = 0
    test_episode_rewards = []
    test_episode_steps = []
    for i in range(test_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = select_action(state, online_net, 0.0, action_dim)  # Greedy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1
            env.render()
            time.sleep(0.7)
        if reward > 0:
            solved += 1
        test_episode_rewards.append(total_reward)
        test_episode_steps.append(steps)
        print(f"Test Episode {i+1}: Steps={steps}, Total Reward={total_reward:.2f}")
    print(f"Solved {solved}/{test_episodes} test cubes.")

    # Save results to CSV
    hyperparams = {
        "SCRAMBLES": SCRAMBLES,
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
    }
    save_results_to_csv(
        train_rewards=episode_rewards,
        train_steps=episode_steps,
        test_rewards=test_episode_rewards,
        test_steps=test_episode_steps,
        test_solved=solved,
        hyperparams=hyperparams,
        output_path=RESULTS_PATH
    )

if __name__ == "__main__":
    main()
