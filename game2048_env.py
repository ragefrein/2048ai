import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class Game2048Env(gym.Env):
    """
    Environment 2048 untuk Reinforcement Learning.
    Aksi:
      0: Geser Kiri (Left)
      1: Geser Atas (Up)
      2: Geser Kanan (Right)
      3: Geser Bawah (Down)
    """
    def __init__(self, render_mode=None):
        super(Game2048Env, self).__init__()
        self.action_space = spaces.Discrete(4)
        # Observasi One-Hot: shape (16, 4, 4) mewakili 16 kemungkinan pangkat nilai 2 (0=kosong, 1=2, 2=4, ..., 15=32768)
        self.observation_space = spaces.Box(low=0, high=1, shape=(16, 4, 4), dtype=np.float32)
        self.render_mode = render_mode
        self.fig, self.ax = None, None
        self.score = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((4, 4), dtype=np.float32)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self._get_obs(), {"valid_actions": self.get_valid_actions(), "score": self.score}

    def _get_obs(self):
        """Representasi One-Hot berdimensi (16, 4, 4)."""
        obs = np.zeros((16, 4, 4), dtype=np.float32)
        powers = np.zeros((4, 4), dtype=np.int64)
        non_zero = self.board > 0
        powers[non_zero] = np.minimum(np.log2(self.board[non_zero]).astype(np.int64), 15)
        np.put_along_axis(obs, powers[None, :, :], 1.0, axis=0)
        return obs

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            idx = np.random.choice(len(empty_cells))
            r, c = empty_cells[idx]
            self.board[r, c] = 2 if np.random.rand() < 0.9 else 4

    def get_valid_actions(self):
        """Mengecek aksi mana saja yang benar-benar mengubah papan (legal moves)."""
        valid = []
        for a in range(4):
            temp_board = self.board.copy()
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
        merge_score, merge_reward = self.slide_and_merge(action)
        self.score += merge_score
        
        # Penalti jika memilih aksi yang tidak valid (papan tidak bergerak)
        if np.array_equal(old_board, self.board):
            reward = -2.0
            valid_actions = self.get_valid_actions()
            terminated = len(valid_actions) == 0
            return self._get_obs(), reward, terminated, False, {
                "valid_actions": valid_actions,
                "score": self.score
            }

        self.add_random_tile()

        valid_actions = self.get_valid_actions()
        terminated = len(valid_actions) == 0 or not self.can_move()

        # --- REWARD SHAPING TERSTRUKTUR ---
        # 1. Log-scale reward penggabungan ubin (stabil, mencegah ledakan gradien)
        reward = merge_reward

        # 2. Bonus petak kosong (mendorong AI menjaga papan tetap lega)
        empty_cells = np.count_nonzero(self.board == 0)
        reward += 0.1 * empty_cells

        # 3. Bonus pojok untuk ubin terbesar (membentuk susunan monotonic)
        max_val = self.board.max()
        corners = [self.board[0, 0], self.board[0, 3], self.board[3, 0], self.board[3, 3]]
        if max_val in corners and max_val >= 32:
            reward += 0.5

        # 4. Penalti saat game over
        if terminated:
            reward -= 5.0

        return self._get_obs(), reward, terminated, False, {
            "valid_actions": valid_actions,
            "score": self.score
        }

    def slide_and_merge(self, action):
        self.board = np.rot90(self.board, action)
        merge_score = 0
        merge_reward = 0.0
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
                    merge_score += merged_val
                    # Skala log2: gabung 2+2=4 -> 2.0, 64+64=128 -> 7.0, 1024+1024=2048 -> 11.0
                    merge_reward += float(np.log2(merged_val))
                    skip = True
                else:
                    new_row.append(row[i])
            new_row += [0] * (4 - len(new_row))
            self.board[r] = np.array(new_row)
        
        self.board = np.rot90(self.board, -action)
        return merge_score, merge_reward

    def can_move(self):
        if np.any(self.board == 0): return True
        for r in range(4):
            for c in range(4):
                if r + 1 < 4 and self.board[r, c] == self.board[r+1, c]: return True
                if c + 1 < 4 and self.board[r, c] == self.board[r, c+1]: return True
        return False

    def render(self):
        if self.render_mode != "human": return

        try:
            if self.fig is None or not plt.fignum_exists(self.fig.number):
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(4, 4))

            self.fig.canvas.manager.set_window_title(f'2048 Visualizer | Score: {int(self.score)} | Max Tile: {int(self.board.max())}')

            self.ax.clear()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_xlim(-0.5, 3.5)
            self.ax.set_ylim(-0.5, 3.5)
            self.ax.invert_yaxis()

            tile_colors = {
                0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
                16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
                256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"
            }

            for r in range(4):
                for c in range(4):
                    val = int(self.board[r, c])
                    text = str(val) if val > 0 else ""
                    color = tile_colors.get(val, "#3c3a32")
                    text_color = "#776e65" if val in [2, 4] else "#f9f6f2"
                    
                    self.ax.add_patch(plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, color=color, ec="#bbada0", lw=2))
                    if text:
                        font_size = 14 if val < 1000 else 11
                        self.ax.text(c, r, text, va='center', ha='center', fontsize=font_size, fontweight='bold', color=text_color)

            plt.draw()
            plt.pause(0.001)
        except Exception:
            # Aman dari crash jika user menutup jendela grafik
            pass

    def close(self):
        if self.fig is not None:
            try:
                plt.close(self.fig)
            except Exception:
                pass
            self.fig, self.ax = None, None