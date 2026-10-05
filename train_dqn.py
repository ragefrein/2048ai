import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os
from game2048_env import Game2048Env

class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.conv_h = nn.Conv2d(16, 128, kernel_size=(1, 2))  # Fitur horizontal
        self.conv_v = nn.Conv2d(16, 128, kernel_size=(2, 1))  # Fitur vertikal
        self.conv_sq = nn.Conv2d(16, 128, kernel_size=(2, 2)) # Fitur 2x2 grid

        in_features = 1536 + 1536 + 1152

        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU()
        )

        self.val_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


        self.adv_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        h = torch.relu(self.conv_h(x))
        v = torch.relu(self.conv_v(x))
        sq = torch.relu(self.conv_sq(x))
        
        flat = torch.cat([h.flatten(1), v.flatten(1), sq.flatten(1)], dim=1)
        feat = self.fc(flat)

        val = self.val_stream(feat)
        adv = self.adv_stream(feat)

        # Dueling formula: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return val + (adv - adv.mean(dim=-1, keepdim=True))



class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Menggunakan perangkat komputasi: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU Terdeteksi: {torch.cuda.get_device_name(0)}")

        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995   
        self.batch_size = 64
        self.update_target_every = 250 
        self.step_counter = 0

        self.model = DuelingDQN().to(self.device)
        self.target_model = DuelingDQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.criterion = nn.SmoothL1Loss() 

    def select_action(self, state, valid_actions=[0, 1, 2, 3]):
        if not valid_actions:
            valid_actions = [0, 1, 2, 3]
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_t = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t).squeeze(0)
        masked_q = torch.full_like(q_values, float('-inf'))
        for a in valid_actions:
            masked_q[a] = q_values[a]

        return torch.argmax(masked_q).item()

    def store_transition(self, state, action, reward, next_state, done, next_valid_actions):
        self.memory.append((state, action, reward, next_state, done, next_valid_actions))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, next_valid_actions = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            online_next_q = self.model(next_states_t)
            mask = torch.zeros((self.batch_size, 4), dtype=torch.bool, device=self.device)
            for i, va in enumerate(next_valid_actions):
                for a in va:
                    mask[i, a] = True
            online_next_q.masked_fill_(~mask, float('-inf'))

            best_next_actions = torch.argmax(online_next_q, dim=1, keepdim=True)
            target_next_q = self.target_model(next_states_t)
            next_q = target_next_q.gather(1, best_next_actions).squeeze(1)

            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.step_counter += 1
        if self.step_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_checkpoint(self, path="best_2048_model.pth"):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, path)

    def load_checkpoint(self, path="best_2048_model.pth"):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.target_model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
            print(f"Berhasil memuat model dari {path}")

if __name__ == "__main__":
    env = Game2048Env(render_mode="human")
    agent = DQNAgent()
    model_save_path = "best_2048_model.pth"
    if os.path.exists(model_save_path):
        agent.load_checkpoint(model_save_path)

    episodes = 1000
    best_max_tile = 0
    best_score = 0

    print("Memulai pelatihan DQN 2048...")

    try:
        for ep in range(episodes):
            state, info = env.reset()
            valid_actions = info["valid_actions"]
            total_reward = 0
            done = False
            visualize_this_ep = (ep % 50 == 0)

            while not done:
                action = agent.select_action(state, valid_actions)
                next_state, reward, done, _, info = env.step(action)
                next_valid_actions = info.get("valid_actions", [])
                agent.store_transition(state, action, reward, next_state, done, next_valid_actions)
                agent.train_step()

                if visualize_this_ep:
                    env.render()

                state = next_state
                valid_actions = next_valid_actions
                total_reward += reward
            agent.decay_epsilon()

            max_tile = int(env.board.max())
            game_score = int(env.score)
            if max_tile > best_max_tile or game_score > best_score:
                best_max_tile = max(best_max_tile, max_tile)
                best_score = max(best_score, game_score)
                agent.save_checkpoint(model_save_path)
                save_msg = " [Model Tersimpan!]"
            else:
                save_msg = ""

            print(f"Episode {ep+1:4d}/{episodes} | Score: {game_score:5d} | Max Tile: {max_tile:4d} | Reward: {total_reward:6.1f} | Epsilon: {agent.epsilon:.3f}{save_msg}")

    except KeyboardInterrupt:
        agent.save_checkpoint(model_save_path)
        print(f"Progress terakhir telah tersimpan ke '{model_save_path}'.")
    finally:
        env.close()