import argparse
import time
import os
import torch
import numpy as np
from collections import Counter
from game2048_env import Game2048Env
from train_dqn import DuelingDQN

def board_to_one_hot(board):
    obs = np.zeros((16, 4, 4), dtype=np.float32)
    powers = np.zeros((4, 4), dtype=np.int64)
    non_zero = board > 0
    powers[non_zero] = np.minimum(np.log2(board[non_zero]).astype(np.int64), 15)
    np.put_along_axis(obs, powers[None, :, :], 1.0, axis=0)
    return obs

def simulate_slide(board, action):
    rotated = np.rot90(board, action)
    merge_reward = 0.0
    new_board = np.zeros((4, 4), dtype=np.float32)
    for r in range(4):
        row = rotated[r][rotated[r] != 0]
        new_row = []
        skip = False
        for i in range(len(row)):
            if skip:
                skip = False
                continue
            if i + 1 < len(row) and row[i] == row[i+1]:
                val = row[i] * 2
                new_row.append(val)
                merge_reward += float(np.log2(val))
                skip = True
            else:
                new_row.append(row[i])
        new_row += [0] * (4 - len(new_row))
        new_board[r] = np.array(new_row)
    new_board = np.rot90(new_board, -action)
    return new_board, merge_reward

def select_action_direct(model, state, valid_actions, device):
    """Pemilihan aksi langsung (Direct DQN) dari model."""
    state_t = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_t).squeeze(0)

    masked_q = torch.full_like(q_values, float('-inf'))
    for a in valid_actions:
        masked_q[a] = q_values[a]

    return torch.argmax(masked_q).item()

def select_action_expectimax(model, current_board, valid_actions, device, gamma=0.99):
    """
    1-Step Expectimax:
    Mensimulasikan papan hasil setiap aksi valid, lalu menghitung
    ekspektasi nilai state probabilistik (90% muncul angka 2, 10% angka 4 di petak kosong).
    """
    if len(valid_actions) == 1:
        return valid_actions[0]

    candidates = []
    weights = []
    action_info = []

    for a in valid_actions:
        next_board, merge_reward = simulate_slide(current_board, a)
        empty_cells = list(zip(*np.where(next_board == 0)))
        start_idx = len(candidates)

        if empty_cells:
            prob = 1.0 / len(empty_cells)
            for r, c in empty_cells:
                # Kemungkinan 1: Muncul angka 2 (probabilitas 90%)
                b2 = next_board.copy()
                b2[r, c] = 2
                candidates.append(board_to_one_hot(b2))
                weights.append(0.9 * prob)

                # Kemungkinan 2: Muncul angka 4 (probabilitas 10%)
                b4 = next_board.copy()
                b4[r, c] = 4
                candidates.append(board_to_one_hot(b4))
                weights.append(0.1 * prob)
        else:
            # Tidak ada petak kosong: cek apakah papan masih bisa bergerak
            # Jika macet, ini posisi berbahaya (game over)
            pass

        end_idx = len(candidates)
        action_info.append((a, merge_reward, start_idx, end_idx, len(empty_cells)))

    if not candidates:
        return valid_actions[0]

    # Evaluasi semua kandidat posisi masa depan dalam 1 batch tensor di GPU
    batch_t = torch.FloatTensor(np.array(candidates)).to(device)
    with torch.no_grad():
        # Ambil nilai Value maksimal dari setiap kemungkinan posisi
        state_values = model(batch_t).max(dim=1)[0]

    weights_t = torch.FloatTensor(weights).to(device)
    weighted_values = state_values * weights_t

    best_action = valid_actions[0]
    best_expected_value = -1e9

    for a, merge_reward, start_idx, end_idx, empty_count in action_info:
        if start_idx == end_idx:
            # Papan penuh sesak tanpa petak kosong
            expected_val = merge_reward - 50.0
        else:
            future_expectation = weighted_values[start_idx:end_idx].sum().item()
            # Nilai gabungan: reward merger langsung + ekspektasi nilai papan ke depan
            # Ditambah sedikit bonus petak kosong agar papan tetap lega
            expected_val = merge_reward + (gamma * future_expectation) + (0.2 * empty_count)

        if expected_val > best_expected_value:
            best_expected_value = expected_val
            best_action = a

    return best_action


def evaluate(model_path="best_2048_model.pth", num_games=10, visualize=False, delay=0.03, mode="expectimax"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perangkat Komputasi : {device}")
    if device.type == "cuda":
        print(f"GPU Terdeteksi      : {torch.cuda.get_device_name(0)}")

    if not os.path.exists(model_path):
        print(f"Error: Model '{model_path}' tidak ditemukan!")
        return

    # Inisialisasi Model & Muat Bobot
    model = DuelingDQN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model Dimuat dari   : {model_path}")
    print(f"Metode Pengambilan  : {'1-Step Expectimax (Search Lookahead)' if mode == 'expectimax' else 'Direct DQN (Greedy Policy)'}\n")

    # Inisialisasi Environment
    render_mode = "human" if visualize else None
    env = Game2048Env(render_mode=render_mode)

    scores = []
    max_tiles = []
    steps_list = []

    print("=" * 70)
    print(f"  MEMULAI EVALUASI AI 2048 ({num_games} GAME | MODE: {mode.upper()})")
    print("=" * 70)

    for g in range(num_games):
        state, info = env.reset()
        valid_actions = info["valid_actions"]
        done = False
        steps = 0

        while not done:
            if mode == "expectimax":
                action = select_action_expectimax(model, env.board, valid_actions, device)
            else:
                action = select_action_direct(model, state, valid_actions, device)

            next_state, _, done, _, info = env.step(action)
            valid_actions = info.get("valid_actions", [])

            if visualize:
                env.render()
                if delay > 0:
                    time.sleep(delay)

            state = next_state
            steps += 1

        game_score = int(env.score)
        max_tile = int(env.board.max())
        scores.append(game_score)
        max_tiles.append(max_tile)
        steps_list.append(steps)

        print(f"Game {g+1:2d}/{num_games} | Skor: {game_score:5d} | Max Tile: {max_tile:4d} | Langkah: {steps:4d}")

    env.close()

    # --- REKAPITULASI STATISTIK ---
    print("\n" + "=" * 70)
    print("  HASIL EVALUASI & STATISTIK PERFORMA")
    print("=" * 70)
    print(f"Total Game Dimainkan : {num_games}")
    print(f"Rata-rata Skor       : {np.mean(scores):.1f}")
    print(f"Skor Tertinggi       : {np.max(scores)}")
    print(f"Skor Terendah        : {np.min(scores)}")
    print(f"Rata-rata Langkah    : {np.mean(steps_list):.1f}")
    print("-" * 70)
    print("Distribusi Pencapaian Max Tile:")
    
    tile_counts = Counter(max_tiles)
    for tile in sorted([64, 128, 256, 512, 1024, 2048, 4096]):
        reach_or_higher = sum(c for t, c in tile_counts.items() if t >= tile)
        pct = (reach_or_higher / num_games) * 100
        bar = "#" * int(pct / 5)
        print(f"  Tile >={tile:4d} : {reach_or_higher:2d}/{num_games} game ({pct:5.1f}%) | {bar}")

    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi AI 2048 DQN")
    parser.add_argument("--games", type=int, default=5, help="Jumlah game yang akan diuji (default: 5)")
    parser.add_argument("--mode", type=str, choices=["expectimax", "direct"], default="expectimax",
                        help="Metode keputusan: 'expectimax' (1-step lookahead) atau 'direct' (default: expectimax)")
    parser.add_argument("--visualize", action="store_true", help="Tampilkan visualisasi GUI pertandingan langsung")
    parser.add_argument("--delay", type=float, default=0.03, help="Jeda antar langkah dalam detik saat visualisasi (default: 0.03)")
    parser.add_argument("--model", type=str, default="best_2048_model.pth", help="Path ke file model checkpoint")

    args = parser.parse_args()
    evaluate(model_path=args.model, num_games=args.games, visualize=args.visualize, delay=args.delay, mode=args.mode)
