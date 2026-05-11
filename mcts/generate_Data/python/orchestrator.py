import argparse
import json
import time
import os
import multiprocessing as mp
from mcts_datagen import MCTSDataGenerator

def worker_task(worker_id: int, games_to_play: int, base_seed: int, num_simulations: int, output_dir: str) -> None:
    """Worker process that writes data sequentially to a file."""

    output_file = os.path.join(output_dir, f"dataset_w{worker_id}.jsonl")
    
    print(f"[Worker {worker_id}] Started. Writing to '{output_file}'.")
    
    with open(output_file, "w") as f:
        for i in range(games_to_play):
            current_seed = base_seed + (worker_id * 100000) + i
            
            generator = MCTSDataGenerator(
                num_simulations=num_simulations, 
                exploration_strength=1.414, 
                random_seed=current_seed
            )
            
            game_data = generator.generate_data()
            
            for step_data in game_data:
                f.write(json.dumps(step_data) + "\n")
            
            f.flush() 
            print(f"[Worker {worker_id}] Game {i+1}/{games_to_play} saved.")
            
            print(f"[Worker {worker_id}] Finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-games", type=int, default=10_000)
    parser.add_argument("--num-simulations", type=int, default=50_000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-folder", default="raw_data")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created folder '{args.output_folder}'.")

    print("=== STARTING PARALLEL GENERATION ===")
    print(f"Target: {args.total_games} games split across {args.num_workers} workers.")
    
    start_time = time.time()

    base_games_per_worker = args.total_games // args.num_workers
    remainder = args.total_games % args.num_workers
    
    tasks = []
    for i in range(args.num_workers):
        games_for_this_worker = base_games_per_worker
        if i == args.num_workers - 1:
            games_for_this_worker += remainder
            
        tasks.append((i, games_for_this_worker, args.base_seed, args.num_simulations, args.output_folder))

    print("Launching CPU workers...\n")
    
    with mp.Pool(processes=args.num_workers) as pool:
        pool.starmap(worker_task, tasks)

    end_time = time.time()
    total_time = end_time - start_time
    print("\n=== DONE ===")
    print(f"All data saved in: {args.output_folder}/")
    print(f"Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()