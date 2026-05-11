import argparse
import os
import json
import glob
import matplotlib.pyplot as plt

def plot_game_lengths(data_dir: str = "raw_data"):
    """Read .jsonl files, compute game lengths, and plot a histogram."""
    game_lengths = []
    
    file_paths = glob.glob(os.path.join(data_dir, "*.jsonl"))
    
    if not file_paths:
        print(f"Warning: no .jsonl files found in '{data_dir}'.")
        return

    print(f"Reading {len(file_paths)} files. This may take a few seconds...")

    for file_path in file_paths:
        previous_step = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                current_step = data.get("step", 0)
                
                if current_step == 1 and previous_step > 0:
                    game_lengths.append(previous_step)
                    
                previous_step = current_step
            
            if previous_step > 0:
                game_lengths.append(previous_step)

    if not game_lengths:
        print("No valid games found in the files.")
        return

    total_games = len(game_lengths)
    mean_len = sum(game_lengths) / total_games
    min_len = min(game_lengths)
    max_len = max(game_lengths)

    print("\n=== DATA SUMMARY ===")
    print(f"Total games read: {total_games}")
    print(f"Average moves: {mean_len:.1f}")
    print(f"Shortest game: {min_len} moves")
    print(f"Longest game: {max_len} moves")

    plt.figure(figsize=(10, 6))
    plt.hist(game_lengths, bins=range(1, 83), edgecolor='black', color='#4C72B0', alpha=0.8)

    plt.title(f'Game Length Distribution (Total: {total_games} games)', fontsize=14, pad=15)
    plt.xlabel('Total Moves', fontsize=12)
    plt.ylabel('Number of Games', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.axvline(mean_len, color='red', linestyle='dashed', linewidth=2, label=f'Average ({mean_len:.1f})')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="raw_data")
    args = parser.parse_args()
    plot_game_lengths(args.data_dir)