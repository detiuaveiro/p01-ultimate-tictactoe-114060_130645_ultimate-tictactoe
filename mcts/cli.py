import argparse
import subprocess
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def run_c_generator(args):
    """Compiles and runs the C MCTS data generator."""
    c_dir = os.path.join(BASE_DIR, "generate_Data", "C")
    
    # Ensure data folder exists
    data_dir = args.output_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(BASE_DIR, data_dir)
    ensure_dir(data_dir)
    
    print("Building C project...")
    try:
        # Use shell=True on Windows to help locate standard 'make' utilities if available
        is_windows = sys.platform == "win32"
        make_process = subprocess.run(["make"], cwd=c_dir, shell=is_windows)
        if make_process.returncode != 0:
            print("Error compiling C project. Exiting.")
            sys.exit(1)
    except FileNotFoundError:
        print("Error: 'make' command not found. On Windows, please ensure you have 'make' installed (via MinGW, MSYS2) or run this script in WSL/Linux.")
        sys.exit(1)
        
    print(f"Running C generator... Outputting to: {data_dir}")
    
    exe_name = "orchestrator.exe" if sys.platform == "win32" else "./orchestrator"
    exe_path = os.path.join(c_dir, exe_name)
    
    # Fallback to without .exe just in case it was built inside a bash-like environment
    if sys.platform == "win32" and not os.path.exists(exe_path) and os.path.exists(os.path.join(c_dir, "orchestrator")):
        exe_path = os.path.join(c_dir, "orchestrator")
        
    cmd = [exe_path]
    cmd.extend(["--total-games", str(args.total_games)])
    cmd.extend(["--num-simulations", str(args.num_simulations)])
    cmd.extend(["--num-workers", str(args.num_workers)])
    cmd.extend(["--seed", str(args.seed)])
    cmd.extend(["--output-dir", data_dir])
    
    subprocess.run(cmd, cwd=c_dir)

def run_py_generator(args):
    """Runs the Python MCTS data generator."""
    py_dir = os.path.join(BASE_DIR, "generate_Data", "python")
    py_script = os.path.join(py_dir, "orchestrator.py")
    
    # Ensure data folder exists
    data_dir = args.output_folder
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(BASE_DIR, data_dir)
    ensure_dir(data_dir)
    
    print(f"Running Python generator... Outputting to: {data_dir}")
    cmd = [sys.executable, py_script]
    cmd.extend(["--total-games", str(args.total_games)])
    cmd.extend(["--num-simulations", str(args.num_simulations)])
    cmd.extend(["--num-workers", str(args.num_workers)])
    cmd.extend(["--base-seed", str(args.base_seed)])
    cmd.extend(["--output-folder", data_dir])
    
    subprocess.run(cmd, cwd=py_dir)

def run_training(args):
    """Runs the PyTorch PVN training script."""
    training_script = os.path.join(BASE_DIR, "policy_value_network", "training.py")
    
    # Ensure directories exist
    save_path = args.save_path
    if not os.path.isabs(save_path):
        save_path = os.path.join(BASE_DIR, save_path)
    ensure_dir(os.path.dirname(save_path))
    
    plot_path = args.plot_path
    if not os.path.isabs(plot_path):
        plot_path = os.path.join(BASE_DIR, plot_path)
    ensure_dir(os.path.dirname(plot_path))
    
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(BASE_DIR, data_path)
    
    print("Starting Training...")
    cmd = [sys.executable, training_script]
    cmd.extend(["--data-path", data_path])
    cmd.extend(["--batch-size", str(args.batch_size)])
    cmd.extend(["--lr", str(args.lr)])
    cmd.extend(["--weight-decay", str(args.weight_decay)])
    cmd.extend(["--num-workers", str(args.num_workers)])
    cmd.extend(["--epochs", str(args.epochs)])
    cmd.extend(["--save-path", save_path])
    cmd.extend(["--device", args.device])
    cmd.extend(["--val-split", str(args.val_split)])
    cmd.extend(["--seed", str(args.seed)])
    cmd.extend(["--early-stop-patience", str(args.early_stop_patience)])
    cmd.extend(["--early-stop-min-delta", str(args.early_stop_min_delta)])
    cmd.extend(["--plot-path", plot_path])
    
    if args.load_model:
        cmd.extend(["--load-model", args.load_model])
    
    subprocess.run(cmd)

def run_visualize(args):
    """Runs the visualization script."""
    vis_script = os.path.join(BASE_DIR, "datatools", "visualize_data.py")
    
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(BASE_DIR, data_dir)
        
    print(f"Starting Visualization for {data_dir}...")
    cmd = [sys.executable, vis_script]
    cmd.extend(["--data-dir", data_dir])
    
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Ultimate Tic-Tac-Toe Project Orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    # --- Subcommand: Generate Data (C) ---
    parser_c = subparsers.add_parser("generateC", help="Generate MCTS data using C program (faster)")
    parser_c.add_argument("--total-games", type=int, default=10000)
    parser_c.add_argument("--num-simulations", type=int, default=50000)
    parser_c.add_argument("--num-workers", type=int, default=8)
    parser_c.add_argument("--seed", type=int, default=42)
    parser_c.add_argument("--output-dir", type=str, default="raw_data", help="Saves inside mcts/raw_data unless absolute")
    parser_c.set_defaults(func=run_c_generator)

    # --- Subcommand: Generate Data (Python) ---
    parser_py = subparsers.add_parser("generatePy", help="Generate MCTS data using Python program")
    parser_py.add_argument("--total-games", type=int, default=10000)
    parser_py.add_argument("--num-simulations", type=int, default=50000)
    parser_py.add_argument("--num-workers", type=int, default=8)
    parser_py.add_argument("--base-seed", type=int, default=42)
    parser_py.add_argument("--output-folder", type=str, default="raw_data", help="Saves inside mcts/raw_data unless absolute")
    parser_py.set_defaults(func=run_py_generator)

    # --- Subcommand: Train Network ---
    parser_train = subparsers.add_parser("train", help="Train the Policy-Value Network")
    parser_train.add_argument("--data-path", default="raw_data")
    parser_train.add_argument('--batch-size', type=int, default=512)
    parser_train.add_argument('--lr', type=float, default=1e-4)
    parser_train.add_argument('--weight-decay', type=float, default=1e-4)
    parser_train.add_argument('--num-workers', type=int, default=0)
    parser_train.add_argument('--epochs', type=int, default=100)
    parser_train.add_argument('--save-path', default='models/modelo_pvn.pth')
    parser_train.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser_train.add_argument('--val-split', type=float, default=0.1)
    parser_train.add_argument('--seed', type=int, default=42)
    parser_train.add_argument('--early-stop-patience', type=int, default=5)
    parser_train.add_argument('--early-stop-min-delta', type=float, default=1e-4)
    parser_train.add_argument('--plot-path', default='plots/training_curves.png')
    parser_train.add_argument('--load-model', default=None, type=str, help="Path to a pre-trained .pth file to load weights from")
    parser_train.set_defaults(func=run_training)

    # --- Subcommand: Visualize Data ---
    parser_vis = subparsers.add_parser("visualize", help="Visualize generated dataset states")
    parser_vis.add_argument("--data-dir", default="raw_data")
    parser_vis.set_defaults(func=run_visualize)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()