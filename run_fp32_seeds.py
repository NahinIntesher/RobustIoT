import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [42, 123, 2026]


def run(command):
    print("\n$ " + " ".join(map(str, command)))
    subprocess.run(command, check=True)


def main(args):
    python = sys.executable

    for model in args.models:
        for seed in args.seeds:
            output = Path("experiments") / f"{model}_fp32" / f"seed_{seed}"

            train_cmd = [
                python,
                "train.py",
                "--model",
                model,
                "--seed",
                str(seed),
                "--split-seed",
                str(args.split_seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--momentum",
                str(args.momentum),
                "--weight-decay",
                str(args.weight_decay),
                "--num-workers",
                str(args.num_workers),
                "--output",
                str(output),
            ]
            run(train_cmd)

            eval_cmd = [
                python,
                "evaluate.py",
                "--checkpoint",
                str(output / "best_model.pth"),
                "--device",
                "auto",
                "--num-workers",
                str(args.num_workers),
            ]
            run(eval_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the paper's repeated FP32 baseline experiments."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tinycnn", "mobilenetv2"],
        choices=["tinycnn", "mobilenetv2"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    main(parser.parse_args())
