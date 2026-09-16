import argparse
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.dataset import build_loaders
from src.model import build_model, canonical_model_name, count_parameters
from src.utils import get_device, save_json, set_seed


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp_path)
    temp_path.replace(path)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    desc="Train",
    show_progress=True,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    progress = tqdm(
        loader,
        desc=f"{desc:<18}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        position=0,
        disable=not show_progress,
    )

    with context:
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_seen += batch_size

            if show_progress:
                progress.set_postfix(
                    loss=f"{total_loss / total_seen:.4f}",
                    acc=f"{100.0 * total_correct / total_seen:.2f}%",
                    refresh=False,
                )

    if total_seen == 0:
        raise RuntimeError(f"{desc} loader produced zero samples.")

    return total_loss / total_seen, total_correct / total_seen


def get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
      try:
          torch_state = state["torch"]

          if isinstance(torch_state, torch.Tensor):
              torch_state = torch_state.cpu().byte()

          torch.set_rng_state(torch_state)

      except Exception as e:
          print(f"Warning: Could not restore torch RNG state: {e}")
          print("Continuing without RNG restore.")
    if torch.cuda.is_available() and "cuda" in state:
        try:
            cuda_states = [
                s.cpu().byte() if isinstance(s, torch.Tensor) else s
                for s in state["cuda"]
            ]
            torch.cuda.set_rng_state_all(cuda_states)

        except Exception as e:
            print(f"Warning: Could not restore CUDA RNG state: {e}")


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_acc, config):
    atomic_torch_save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config,
            "rng_state": get_rng_state(),
        },
        Path(path),
    )


def save_best_model(path, epoch, model, best_val_acc, config):
    atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "config": config,
        },
        Path(path),
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    print("\n" + "=" * 66)
    print("CHECKPOINT FOUND")
    print("=" * 66)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint.get("rng_state"))

    last_epoch = int(checkpoint["epoch"])
    best_val_acc = float(checkpoint.get("best_val_acc", -1.0))
    start_epoch = last_epoch + 1

    print(f"Last completed epoch : {last_epoch}")
    print(f"Resume from epoch    : {start_epoch}")
    print(f"Best validation acc  : {best_val_acc * 100:.2f}%")
    print("=" * 66 + "\n")
    return start_epoch, best_val_acc, checkpoint.get("config", {})


def validate_resume_config(saved_config: dict, current_config: dict) -> None:
    if not saved_config:
        return

    critical_keys = ["model", "seed", "split_seed", "val_size"]
    mismatches = []
    for key in critical_keys:
        if key in saved_config and saved_config[key] != current_config.get(key):
            mismatches.append(
                f"{key}: checkpoint={saved_config[key]!r}, current={current_config.get(key)!r}"
            )

    if mismatches:
        joined = "\n  - ".join(mismatches)
        raise RuntimeError(
            "Refusing to resume because critical experiment settings changed:\n"
            f"  - {joined}\n"
            "Use a different --output directory or pass --no-resume for a fresh run."
        )


def plot_history(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train")
    plt.plot(df["epoch"], df["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="Train")
    plt.plot(df["epoch"], df["val_acc"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy.png", dpi=160)
    plt.close()


def load_history(log_path: Path) -> list:
    if not log_path.exists():
        return []
    try:
        history = pd.read_csv(log_path).to_dict("records")
        print(f"Existing training log: {len(history)} epochs")
        return history
    except Exception as exc:
        print(f"Warning: could not load existing training log: {exc}")
        return []


def mean_recent_epoch_time(history: list, session_times: list):
    if session_times:
        return float(np.mean(session_times[-5:]))

    historical = []
    for row in history:
        value = row.get("epoch_time_sec")
        if value is None or pd.isna(value):
            continue
        historical.append(float(value))
    return float(np.mean(historical[-5:])) if historical else None


def remove_fresh_run_artifacts(paths) -> None:
    for path in paths:
        path = Path(path)
        if path.exists() and path.is_file():
            path.unlink()


def main(args):
    args.model = canonical_model_name(args.model)
    set_seed(args.seed)
    device = get_device(args.device)

    if args.output is None:
        out = Path("experiments") / f"{args.model}_fp32" / f"seed_{args.seed}"
    else:
        out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = out / "last_checkpoint.pth"
    best_checkpoint_path = out / "best_model.pth"
    log_path = out / "train_log.csv"
    plots_dir = out / "plots"

    config = vars(args).copy()
    config["output"] = str(out)
    config["device_resolved"] = str(device)

    train_loader, val_loader, _ = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
        val_size=args.val_size,
    )

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    print("=" * 66)
    print("RobustQuant-IoT | FP32 Baseline")
    print("=" * 66)
    print(f"Model                : {args.model}")
    print(f"Device               : {device}")
    if device.type == "cuda":
        print(f"GPU                  : {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA         : {torch.version.cuda}")
    print(f"Trainable parameters : {count_parameters(model):,}")
    print(f"Run seed             : {args.seed}")
    print(f"Fixed split seed     : {args.split_seed}")
    print(f"Target epochs        : {args.epochs}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Initial LR           : {args.lr}")
    print(f"Output               : {out.resolve()}")

    start_epoch = 1
    best_val = -1.0
    history = load_history(log_path)

    if args.no_resume:
        print("\nAuto-resume disabled. Starting a clean run.\n")
        remove_fresh_run_artifacts(
            [last_checkpoint_path, best_checkpoint_path, log_path]
        )
        history = []
    elif last_checkpoint_path.exists():
        start_epoch, best_val, saved_config = load_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            device,
        )
        validate_resume_config(saved_config, config)
        history = [
            row for row in history if int(row["epoch"]) < start_epoch
        ]
    else:
        print("\nNo checkpoint found. Starting training from epoch 1.\n")
        # Prevent stale artifacts from being mistaken for this run.
        remove_fresh_run_artifacts([best_checkpoint_path, log_path])
        history = []

    save_json(config, out / "config.json")

    if start_epoch > args.epochs:
        print(f"Training already completed through epoch {start_epoch - 1}.")
        if history:
            plot_history(pd.DataFrame(history), plots_dir)
        return

    session_start = time.perf_counter()
    session_epoch_times = []

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            synchronize_device(device)
            epoch_start = time.perf_counter()

            print("\n")
            print("=" * 70)
            print(f"                EPOCH {epoch}/{args.epochs}")
            print("=" * 70)

            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                desc=f"Train {epoch}/{args.epochs}",
                show_progress=not args.no_progress,
            )
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                desc=f"Val   {epoch}/{args.epochs}",
                show_progress=not args.no_progress,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            is_new_best = val_acc > best_val
            if is_new_best:
                best_val = val_acc
                save_best_model(
                    best_checkpoint_path,
                    epoch,
                    model,
                    best_val,
                    config,
                )

            scheduler.step()
            next_lr = optimizer.param_groups[0]["lr"]

            synchronize_device(device)
            epoch_time = time.perf_counter() - epoch_start
            session_epoch_times.append(epoch_time)
            session_elapsed = time.perf_counter() - session_start

            estimated_epoch_time = mean_recent_epoch_time(
                history, session_epoch_times
            )
            remaining_epochs = args.epochs - epoch
            eta = (
                estimated_epoch_time * remaining_epochs
                if estimated_epoch_time is not None
                else None
            )

            row = {
                "epoch": epoch,
                "lr": current_lr,
                "next_lr": next_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val,
                "epoch_time_sec": epoch_time,
            }
            history = [h for h in history if int(h["epoch"]) != epoch]
            history.append(row)
            history.sort(key=lambda item: int(item["epoch"]))

            pd.DataFrame(history).to_csv(log_path, index=False)

            if args.plot_every > 0 and (
                epoch % args.plot_every == 0 or epoch == args.epochs
            ):
                plot_history(pd.DataFrame(history), plots_dir)

            save_checkpoint(
                last_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                best_val,
                config,
            )

            print(f"Train Loss   : {train_loss:.4f}")
            print(f"Train Acc    : {train_acc * 100:.2f}%")
            print(f"Val Loss     : {val_loss:.4f}")
            print(f"Val Acc      : {val_acc * 100:.2f}%")
            print(f"Best Val     : {best_val * 100:.2f}%")
            print(f"LR used      : {current_lr:.8f}")
            print(f"Next LR      : {next_lr:.8f}")
            print(f"Epoch Time   : {format_duration(epoch_time)}")
            print(f"Session Time : {format_duration(session_elapsed)}")
            print(
                f"ETA          : {format_duration(eta)}"
                if eta is not None
                else "ETA          : calculating..."
            )
            if is_new_best:
                print("✓ New best model saved")
            print("✓ Resume checkpoint saved")
            print("✓ CSV log updated")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        print(
            "The last fully completed epoch is stored in:\n"
            f"{last_checkpoint_path.resolve()}"
        )
        print("Run the same command again to resume.")
        return

    plot_history(pd.DataFrame(history), plots_dir)
    total_session_time = time.perf_counter() - session_start

    print("\n" + "=" * 66)
    print("TRAINING COMPLETE")
    print("=" * 66)
    print(f"Best validation accuracy : {best_val * 100:.2f}%")
    print(f"Session training time    : {format_duration(total_session_time)}")
    print(f"Best model               : {best_checkpoint_path.resolve()}")
    print(f"Last checkpoint          : {last_checkpoint_path.resolve()}")
    print(f"Training log             : {log_path.resolve()}")
    print(f"Plots                    : {plots_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "FP32 training for the RobustQuant-IoT TinyCNN/MobileNetV2 baselines."
        )
    )
    parser.add_argument("--model", default="tinycnn", choices=["tinycnn", "mobilenetv2"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--output",
        default=None,
        help="Default: experiments/<model>_fp32/seed_<seed>",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training/run seed. Change this for independent repeated runs.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed train/validation split seed. Keep unchanged across all runs.",
    )
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    main(parser.parse_args())
