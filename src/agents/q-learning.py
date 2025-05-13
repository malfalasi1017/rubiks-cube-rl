import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from gymnasium_env.env.rubiks_cube import RubiksCubeEnv

class QlearningAgent():
    def __init__(
        self,
        env,
        learning_rate=0.1,
        initial_epsilon=1.0,
        epsilon_decay=0.99,
        final_epsilon=0.1,
        discount_factor=0.9,
    ):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.episode_rewards = []
        self.episode_steps = []

    def select_action(self, env, state):
        state = tuple(state)
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.q_values[state])

    def update(self, state, action, reward, next_state):
        state = tuple(state)
        next_state = tuple(next_state)
        future_q = np.max(self.q_values[next_state])
        td = reward + self.discount_factor * future_q - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td
        self.training_error.append(td)
        self.training_error = self.training_error[-1000:]

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(dict(self.q_values), f)

    def load_q_table(self, file_path):
        with open(file_path, 'rb') as f:
            self.q_values = defaultdict(lambda: np.zeros(len(next(iter(self.q_values.values())))), pickle.load(f))

def main():
    env = RubiksCubeEnv(scrambles=4, max_steps=30)
    agent = QlearningAgent(
        env=env,
        learning_rate=0.1,
        initial_epsilon=1.0,
        epsilon_decay=0.995,
        final_epsilon=0.05,
        discount_factor=0.95,
    )

    print("Starting training...\n")
    solved_count = 0
    for episode in tqdm(range(1, 200_00)):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(env, state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update(state, action, reward, next_state)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1

        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(steps)
        agent.decay_epsilon()
        if reward > 0:
            solved_count += 1

        if episode % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_steps = np.mean(agent.episode_steps[-100:])
            print(f"Episode {episode:4d} | Avg Reward (last 100): {avg_reward:.3f} | "
                  f"Avg Steps: {avg_steps:.2f} | Epsilon: {agent.epsilon:.3f} | "
                  f"Solved (last 100): {solved_count}/100")
            solved_count = 0

    agent.save_q_table("q_table.pkl")
    print("\nTraining complete. Q-table saved as q_table.pkl\n")

    # Plotting training statistics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(agent.episode_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(agent.episode_steps, label='Episode Steps', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(agent.training_error, label='Training Error')
    plt.xlabel('Update Step')
    plt.ylabel('TD Error')
    plt.title('Training Error Over Time')
    plt.legend()
    plt.show()

    # Testing the agent
    print("\nTesting the trained agent...\n")
    test_episodes = 10
    solved = 0
    for i in range(test_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        print(f"\nTest Episode {i+1}")
        while not done:
            action = np.argmax(agent.q_values[tuple(state)])  # Greedy action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1
            print(f"  Step {steps:2d}: Action={action}, Reward={reward:.2f}, Done={done}")
            env.render()
        if reward > 0:
            solved += 1
        print(f"Episode {i+1} finished in {steps} steps with total reward {total_reward:.2f}")

    print(f"\nTest Results: Solved {solved}/{test_episodes} cubes.")

if __name__ == "__main__":
    main()