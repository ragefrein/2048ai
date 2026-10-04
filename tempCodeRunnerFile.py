import torch
import torch.nn as nn
import torch.optim as optim
import random
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
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0 # Eksplorasi awal 100%
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        
        self.model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
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
    # Inisialisasi env dengan render_mode='human'
    env = Game2048Env(render_mode="human")
    agent = DQNAgent()
    episodes = 1000

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        # Tampilkan visualisasi hanya pada episode kelipatan 20 atau episode terakhir
        visualize_this_ep = (ep % 20 == 0)

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            if visualize_this_ep:
                env.render()

            state = next_state
            total_reward += reward

        if ep == episodes - 1:
            env.render()

        max_tile = int(2 ** env.board.max()) if env.board.max() > 0 else 0
        print(f"Episode {ep+1}/{episodes} - Reward: {total_reward} - Max Tile: {max_tile}")