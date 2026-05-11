import argparse
import math
import random
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Ensure the root project directory is in the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mcts.policy_value_network.pvn import PVN
from mcts.datatools.dataset_pp import UTTTDataset
from mcts.policy_value_network.pvn_loss import alpha_zero_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='raw_data')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save-path', default='models/modelo_pvn.pth')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early-stop-patience', type=int, default=5)
    parser.add_argument('--early-stop-min-delta', type=float, default=1e-4)
    parser.add_argument('--plot-path', default='plots/training_curves.png')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = PVN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    base_dataset = UTTTDataset(args.data_path, augment=True)
    total_size = len(base_dataset)
    val_size = int(total_size * args.val_split)
    if val_size < 1 or total_size < 2:
        raise ValueError("val-split is too small to create a validation set.")
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(base_dataset, train_indices)
    val_base_dataset = UTTTDataset(args.data_path, data_lines=base_dataset.data_lines, augment=False)
    val_dataset = Subset(val_base_dataset, val_indices)

    print(f"Train: {train_size} samples | Validation: {val_size} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == 'cuda'
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR schedule: warm up for 5 epochs, then cosine decay
    total_steps = args.epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    print("\n--- STARTING TRAINING ---")

    history = {
        "train_total": [],
        "train_policy": [],
        "train_value": [],
        "val_total": [],
        "val_policy": [],
        "val_value": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    prev_train_loss = None
    prev_val_loss = None

    for epoch in range(args.epochs):
        total_loss = total_p_loss = total_v_loss = 0
        model.train()

        for batch_idx, (inputs, targets_pi, targets_v, policy_mask) in enumerate(train_loader):
            inputs      = inputs.to(device)
            targets_pi  = targets_pi.to(device)
            targets_v   = targets_v.to(device)
            policy_mask = policy_mask.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                policy_preds, value_preds = model(inputs)
                loss, p_loss, v_loss = alpha_zero_loss(
                    policy_preds, value_preds, targets_pi, targets_v, policy_mask
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss   += loss.item()
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()

            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

        avg_loss   = total_loss   / len(train_loader)
        avg_p_loss = total_p_loss / len(train_loader)
        avg_v_loss = total_v_loss / len(train_loader)

        history["train_total"].append(avg_loss)
        history["train_policy"].append(avg_p_loss)
        history["train_value"].append(avg_v_loss)

        # Validation
        model.eval()
        val_total = val_p_total = val_v_total = 0
        with torch.no_grad():
            for inputs, targets_pi, targets_v, policy_mask in val_loader:
                inputs      = inputs.to(device)
                targets_pi  = targets_pi.to(device)
                targets_v   = targets_v.to(device)
                policy_mask = policy_mask.to(device)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    policy_preds, value_preds = model(inputs)
                    loss, p_loss, v_loss = alpha_zero_loss(
                        policy_preds, value_preds, targets_pi, targets_v, policy_mask
                    )

                val_total   += loss.item()
                val_p_total += p_loss.item()
                val_v_total += v_loss.item()

        val_avg_loss   = val_total   / len(val_loader)
        val_avg_p_loss = val_p_total / len(val_loader)
        val_avg_v_loss = val_v_total / len(val_loader)

        history["val_total"].append(val_avg_loss)
        history["val_policy"].append(val_avg_p_loss)
        history["val_value"].append(val_avg_v_loss)

        print(f"\n>>> END OF EPOCH {epoch+1}")
        print(
            f"Train Loss: {avg_loss:.4f} | Policy: {avg_p_loss:.4f} | Value: {avg_v_loss:.4f}\n"
            f"Val   Loss: {val_avg_loss:.4f} | Policy: {val_avg_p_loss:.4f} | Value: {val_avg_v_loss:.4f}\n"
        )

        # Save best model by validation loss
        if val_avg_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_avg_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.save_path)

        # Early stopping: validation rises while training falls
        if prev_train_loss is not None and prev_val_loss is not None:
            train_decreasing = avg_loss < prev_train_loss - args.early_stop_min_delta
            val_increasing = val_avg_loss > prev_val_loss + args.early_stop_min_delta
            if train_decreasing and val_increasing:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= args.early_stop_patience:
                print(
                    "Early stopping triggered: validation rising while training falls. "
                    f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})."
                )
                break

        prev_train_loss = avg_loss
        prev_val_loss = val_avg_loss

    # Plot
    try:
        import matplotlib.pyplot as plt

        epochs_ran = range(1, len(history["train_total"]) + 1)
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

        axes[0].plot(epochs_ran, history["train_total"], label="train")
        axes[0].plot(epochs_ran, history["val_total"], label="val")
        axes[0].set_ylabel("loss total")
        axes[0].legend()

        axes[1].plot(epochs_ran, history["train_policy"], label="train")
        axes[1].plot(epochs_ran, history["val_policy"], label="val")
        axes[1].set_ylabel("policy loss")
        axes[1].legend()

        axes[2].plot(epochs_ran, history["train_value"], label="train")
        axes[2].plot(epochs_ran, history["val_value"], label="val")
        axes[2].set_ylabel("value loss")
        axes[2].set_xlabel("epoch")
        axes[2].legend()

        fig.tight_layout()
        plt.savefig(args.plot_path, dpi=150)
        print(f"Plot saved to: {args.plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == '__main__':
    main()