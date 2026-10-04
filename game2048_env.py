import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class Game2048Env(gym.Env):
    def __init__(self, render_mode=None):
        super(Game2048Env, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=16, shape=(4, 4), dtype=np.float32)
        self.render_mode = render_mode
        self.fig, self.ax = None, None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((4, 4), dtype=np.float32)
        self.add_random_tile()
        self.add_random_tile()
        return self._get_obs(), {"valid_actions": self.get_valid_actions()}

    def _get_obs(self):
        obs = np.zeros((4, 4), dtype=np.float32)
        non_zero = self.board > 0
        obs[non_zero] = np.log2(self.board[non_zero])
        return obs

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            idx = np.random.choice(len(empty_cells))
            r, c = empty_cells[idx]
            self.board[r, c] = 2 if np.random.rand() < 0.9 else 4

    def get_valid_actions(self):
        """Mengecek aksi mana saja (0:Up, 1:Right, 2:Down, 3:Left) yang benar-benar mengubah papan."""
        valid = []
        for a in range(4):
            temp_board = self.board.copy()
            # Coba gerakkan di papan sementara
            temp_board = np.rot90(temp_board, a)
            moved = False
            for r in range(4):
                row = temp_board[r][temp_board[r] != 0]
                new_row = []
                skip = False
                for i in range(len(row)):
                    if skip:
                        skip = False
                        continue
                    if i + 1 < len(row) and row[i] == row[i+1]:
                        new_row.append(row[i] * 2)
                        skip = True
                    else:
                        new_row.append(row[i])
                new_row += [0] * (4 - len(new_row))
                if not np.array_equal(temp_board[r], new_row):
                    moved = True
                    break
            if moved:
                valid.append(a)
        return valid

    def step(self, action):
        old_board = self.board.copy()
        reward = self.slide_and_merge(action)
        
        if np.array_equal(old_board, self.board):
            reward = -2
        else:
            self.add_random_tile()

        # Bonus kecil jika ubin terbesar bertahan di salah satu sudut papan
        max_pos = np.unravel_index(np.argmax(self.board), self.board.shape)
        if max_pos in [(0, 0), (0, 3), (3, 0), (3, 3)]:
            reward += 1

        valid_actions = self.get_valid_actions()
        terminated = len(valid_actions) == 0 or not self.can_move()

        return self._get_obs(), reward, terminated, False, {"valid_actions": valid_actions}

    def slide_and_merge(self, action):
        self.board = np.rot90(self.board, action)
        reward = 0
        for r in range(4):
            row = self.board[r][self.board[r] != 0]
            new_row = []
            skip = False
            for i in range(len(row)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(row) and row[i] == row[i+1]:
                    merged_val = row[i] * 2
                    new_row.append(merged_val)
                    reward += merged_val
                    skip = True
                else:
                    new_row.append(row[i])
            new_row += [0] * (4 - len(new_row))
            self.board[r] = np.array(new_row)
        
        self.board = np.rot90(self.board, -action)
        return reward

    def can_move(self):
        if np.any(self.board == 0): return True
        for r in range(4):
            for c in range(4):
                if r + 1 < 4 and self.board[r, c] == self.board[r+1, c]: return True
                if c + 1 < 4 and self.board[r, c] == self.board[r, c+1]: return True
        return False

    def render(self):
        if self.render_mode != "human": return

        if self.fig is None or not plt.fignum_exists(self.fig.number):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(4, 4))
            self.fig.canvas.manager.set_window_title('2048 RL Training Visualizer')

        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, 3.5)
        self.ax.set_ylim(-0.5, 3.5)
        self.ax.invert_yaxis()

        for r in range(4):
            for c in range(4):
                val = int(self.board[r, c])
                text = str(val) if val > 0 else ""
                color = "#ccc0b4" if val == 0 else "#eee4da" if val == 2 else "#edc22e" if val >= 1024 else "#f2b179"
                
                self.ax.add_patch(plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, color=color, ec="black"))
                if text:
                    self.ax.text(c, r, text, va='center', ha='center', fontsize=14, fontweight='bold')

        plt.draw()
        plt.pause(0.001)