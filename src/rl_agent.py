import numpy as np
import random
import matplotlib.pyplot as plt
from simulation import TrafficSimulation
import pickle
import os

# Q-Learning Parameters
ALPHA = 0.1  # Learning Rate
GAMMA = 0.9  # Discount Factor
EPSILON = 0.1 # Exploration Rate
EPISODES = 50 # Increase this for better training
STEPS_PER_EPISODE = 200

class QLearningAgent:
    def __init__(self, action_space_size=2):
        self.q_table = {} # State -> [Q_value_action_0, Q_value_action_1]
        self.action_space_size = action_space_size

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < EPSILON:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.get_q_values(state))

    def update(self, state, action, reward, next_state):
        old_q = self.get_q_values(state)[action]
        next_max_q = np.max(self.get_q_values(next_state))
        
        # Q-Learning Formula
        new_q = old_q + ALPHA * (reward + GAMMA * next_max_q - old_q)
        self.q_table[state][action] = new_q

def train_agent():
    sim = TrafficSimulation()
    agent = QLearningAgent()
    
    rewards_per_episode = []
    
    print(f"Starting Training for {EPISODES} Episodes...")
    
    for episode in range(EPISODES):
        # Reset Simulation (Manual reset as TrafficSimulation lacks reset method, simple re-init)
        sim = TrafficSimulation() 
        state = sim.get_state()
        total_reward = 0
        
        for step in range(STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            
            # Apply Action
            sim.step(action)
            
            # Get New State and Reward
            next_state = sim.get_state()
            reward = sim.get_reward()
            
            # Update Q-Table
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
        rewards_per_episode.append(total_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{EPISODES} - Total Reward: {total_reward}")

    # Save Q-Table (Optional)
    # with open('q_table.pkl', 'wb') as f:
    #     pickle.dump(agent.q_table, f)
        
    # Plot Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPISODES + 1), rewards_per_episode, marker='o', linestyle='-')
    plt.title('RL Agent Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Negative Queue Length)')
    plt.grid(True)
    plt.savefig('rl_training_rewards.png')
    print("Training Complete. Reward graph saved to 'rl_training_rewards.png'.")

if __name__ == "__main__":
    train_agent()
