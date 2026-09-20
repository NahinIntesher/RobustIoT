import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))
    
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.model import build_model


# ============================================================
# CIFAR-10 normalization
# Same normalization used by the existing project
# ============================================================
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


# ============================================================
# Corruptions to evaluate first
# ============================================================
CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "motion_blur",
    "defocus_blur",
    "brightness",
    "contrast",
    "jpeg_compression",
]


# ============================================================
# Helpers
# ============================================================
def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return model


def load_cifar10c(cifar10c_dir, corruption, severity):
    """
    CIFAR-10-C stores 50,000 images per corruption.
    Each severity contains 10,000 consecutive images.

    severity:
        1 -> indices 0:10000
        2 -> indices 10000:20000
        3 -> indices 20000:30000
        4 -> indices 30000:40000
        5 -> indices 40000:50000
    """

    path = Path(cifar10c_dir) / f"{corruption}.npy"
    labels_path = Path(cifar10c_dir) / "labels.npy"

    if not path.exists():
        raise FileNotFoundError(f"Missing corruption file: {path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    images = np.load(path)
    labels = np.load(labels_path)

    start = (severity - 1) * 10000
    end = severity * 10000

    images = images[start:end]
    labels = labels[start:end]

    return images, labels


def preprocess(images):
    """
    CIFAR-10-C images are uint8 HWC.
    Convert to normalized float tensor in CHW format.
    """

    images = images.astype(np.float32) / 255.0

    images = (images - MEAN[None, None, :]) / STD[None, None, :]

    images = np.transpose(images, (0, 3, 1, 2))

    return torch.from_numpy(images)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    all_predictions = []
    all_targets = []

    start_time = time.perf_counter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    elapsed = time.perf_counter() - start_time

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    accuracy = accuracy_score(targets, predictions)

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "num_samples": int(len(targets)),
        "evaluation_time_sec": float(elapsed),
    }


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a model on CIFAR-10-C."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["tinycnn", "mobilenetv2"],
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to FP32/QAT/INT8 checkpoint.",
    )

    parser.add_argument(
        "--cifar10c-dir",
        default="./data/CIFAR-10-C",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("RobustQuant-IoT | CIFAR-10-C Robustness Evaluation")
    print("=" * 65)
    print(f"Model       : {args.model}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Dataset     : {args.cifar10c_dir}")
    print(f"Device      : {device}")
    print("=" * 65)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = build_model(args.model, num_classes=10)

    model = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    results = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "dataset": "CIFAR-10-C",
        "device": str(device),
        "corruptions": {},
    }

    for corruption in CORRUPTIONS:

        print()
        print("-" * 65)
        print(f"Corruption: {corruption}")
        print("-" * 65)

        results["corruptions"][corruption] = {}

        for severity in range(1, 6):

            print(f"Severity {severity}/5 ... ", end="", flush=True)

            images, labels = load_cifar10c(
                args.cifar10c_dir,
                corruption,
                severity,
            )

            images = preprocess(images)

            labels = torch.from_numpy(labels).long()

            dataset = TensorDataset(images, labels)

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            metrics = evaluate(
                model,
                loader,
                device,
            )

            results["corruptions"][corruption][str(severity)] = metrics

            print(
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"Macro-F1={metrics['macro_f1']:.4f}"
            )

    # --------------------------------------------------------
    # Mean corruption accuracy
    # --------------------------------------------------------
    all_accuracies = []

    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            value = results["corruptions"][corruption][str(severity)][
                "accuracy"
            ]
            all_accuracies.append(value)

    results["summary"] = {
        "mean_accuracy_all_corruptions_severities": float(
            np.mean(all_accuracies)
        )
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    if args.output is None:

        checkpoint_path = Path(args.checkpoint)

        output_dir = checkpoint_path.parent / "cifar10c"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / "cifar10c_robustness_results.json"
        )

    else:

        output_path = Path(args.output)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 65)
    print("DONE")
    print("=" * 65)
    print(
        "Mean Accuracy "
        f"= {results['summary']['mean_accuracy_all_corruptions_severities']:.4f}"
    )
    print(f"Saved: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()