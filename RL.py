from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os

class RLAgent:
    def __init__(self, n_features, n_actions, alpha=0.1,gamma=0.9, epsilon=0.2):
        self.q_table=np.zeros((n_features, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
    
    def choose_action (self, state_index):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        return np.argmax(self.q_table[state_index])
    
    def update(self, state_index, action, reward):
        current = self.q_table[state_index, action]
        self.q_table[state_index,action] += self.alpha*(reward - current)

def train_agent(episodes=100):
    data = load_iris()
    x = data.data
    y = data.target
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    states = x_scaled
    label = y
    n_action = len(np.unique(y))
    
    agent = RLAgent(n_features=len(states), n_actions = n_action)
    rewards = []
    
    for episode in range(episodes):
        total_reward = 0
        
        for i, state in  enumerate(states):
            state_index = i 
            true_label = label[i]
            action = agent.choose_action(state_index)
            reward = 1 if action == true_label else -1
            agent.update(state_index, action, reward)
            total_reward += reward
            
        rewards.append(total_reward)
    
    if not os.path.exists("static"):
        os.makedirs("static")
    
    plt.figure()
    plt.plot(rewards)
    plt.title("Recompensas por epoca")
    plt.xlabel("Epoca")
    plt.ylabel("Recompensa total")
    plt.grid()
    plt.savefig("static/reward_plot.png")
    plt.close()
    
    correct = sum(np.argmax(agent.q_table[i]) == label[i] for i in range(len(states)))
    accuracy = correct / len(states)

    q_display = [{"index": i, "q_values": list(agent.q_table[i])} for i in range(10)]

    return (accuracy*100),q_display