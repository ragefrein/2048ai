import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from game2048_env import Game2048Env

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998 
        self.batch_size = 64
        
        self.model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()

    def select_action(self, state, valid_actions=[0, 1, 2, 3]):
        if not valid_actions:
            valid_actions = [0, 1, 2, 3]

        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        state_t = torch.FloatTensor(np.array(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t).squeeze(0)
        
        masked_q = torch.full_like(q_values, float('-inf'))
        for a in valid_actions:
            masked_q[a] = q_values[a]
            
        return torch.argmax(masked_q).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones)

        current_q = self.model(states_t).gather(1, actions_t).squeeze(1)
        next_q = self.model(next_states_t).max(1)[0]
        target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = Game2048Env(render_mode="human")
    agent = DQNAgent()
    episodes = 1000

    for ep in range(episodes):
        state, info = env.reset()
        valid_actions = info["valid_actions"]
        total_reward = 0
        done = False
        
        visualize_this_ep = (ep % 20 == 0)

        while not done:
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, _, info = env.step(action)
            valid_actions = info.get("valid_actions", [0, 1, 2, 3])

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            if visualize_this_ep:
                env.render()

            state = next_state
            total_reward += reward

        max_tile = int(env.board.max())
        print(f"Episode {ep+1}/{episodes} - Reward: {total_reward:.1f} - Max Tile: {max_tile} - Epsilon: {agent.epsilon:.2f}")