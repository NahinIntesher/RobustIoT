import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import statistics
import time

import numpy as np
import torch
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from src.dataset import build_test_loader
from src.model import build_model
from src.utils import save_json, set_seed


def load_checkpoint(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "qat_state_dict" in checkpoint:
        state_dict = checkpoint["qat_state_dict"]
    else:
        raise KeyError(
            "Checkpoint does not contain 'state_dict' or 'qat_state_dict'."
        )

    model_name = checkpoint.get(
        "model_name",
        "mobilenetv2",
    )

    return checkpoint, model_name, state_dict


def rebuild_qat_int8_model(model_name, state_dict):
    torch.backends.quantized.engine = "x86"

    model = build_model(
        model_name,
        num_classes=10,
    )

    model.eval()

    qconfig_mapping = tq.QConfigMapping().set_global(
        tq.get_default_qat_qconfig("x86")
    )

    example_inputs = (
        torch.randn(1, 3, 32, 32),
    )

    model = prepare_qat_fx(
        model,
        qconfig_mapping,
        example_inputs,
    )

    model.eval()

    model = convert_fx(model)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


@torch.inference_mode()
def predict(model, loader):
    y_true = []
    y_pred = []

    progress = tqdm(
        loader,
        desc="Test ",
        unit="batch",
    )

    for images, targets in progress:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        y_true.extend(targets.tolist())
        y_pred.extend(predictions.tolist())

        progress.set_postfix(
            acc=f"{100.0 * accuracy_score(y_true, y_pred):.2f}%"
        )

    return np.asarray(y_true), np.asarray(y_pred)


@torch.inference_mode()
def measure_latency(model, warmup=50, runs=300):
    model.eval()

    x = torch.randn(
        1,
        3,
        32,
        32,
    )

    for _ in range(warmup):
        model(x)

    times_ms = []

    for _ in tqdm(
        range(runs),
        desc="Latency",
        unit="run",
    ):
        start = time.perf_counter()
        model(x)
        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        times_ms.append(elapsed)

    mean_ms = statistics.mean(times_ms)

    return {
        "latency_mean_ms": float(mean_ms),
        "latency_median_ms": float(
            statistics.median(times_ms)
        ),
        "latency_p95_ms": float(
            np.percentile(times_ms, 95)
        ),
        "latency_p99_ms": float(
            np.percentile(times_ms, 99)
        ),
        "throughput_inf_per_s": float(
            1000.0 / mean_ms
        ),
    }


def main(args):
    set_seed(args.seed)

    checkpoint_path = Path(
        args.checkpoint
    )

    print("=" * 60)
    print("INT8 QAT Evaluation")
    print("=" * 60)
    print(f"Checkpoint : {checkpoint_path}")
    print("Device     : CPU")

    checkpoint, model_name, state_dict = load_checkpoint(
        checkpoint_path
    )

    print(f"Model      : {model_name}")
    print("Loading INT8 QAT model...")

    model = rebuild_qat_int8_model(
        model_name,
        state_dict,
    )

    test_loader = build_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("\nRunning test evaluation...\n")

    y_true, y_pred = predict(
        model,
        test_loader,
    )

    metrics = {
        "precision_type": "INT8_QAT",
        "model": model_name,
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "model_size_mb": float(
            checkpoint_path.stat().st_size
            / (1024 ** 2)
        ),
    }

    print("\nMeasuring CPU latency...\n")

    metrics.update(
        measure_latency(
            model,
            warmup=args.warmup,
            runs=args.runs,
        )
    )

    output_dir = checkpoint_path.parent

    save_json(
        metrics,
        output_dir / "qat_int8_metrics.json",
    )

    save_json(
        classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        output_dir / "classification_report.json",
    )

    save_json(
        {
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
            ).tolist()
        },
        output_dir / "confusion_matrix.json",
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print()
    print("Saved:")
    print(
        (
            output_dir / "qat_int8_metrics.json"
        ).resolve()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=300)

    main(parser.parse_args())