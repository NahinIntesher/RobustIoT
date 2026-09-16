import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from src.dataset import build_test_loader
from src.model import build_model, canonical_model_name, count_parameters
from src.utils import get_device, save_json, set_seed


@torch.inference_mode()
def predict(model, loader, device):
    targets_all = []
    predictions_all = []

    model.eval()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)

        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()

        targets_all.extend(targets.tolist())
        predictions_all.extend(predictions.tolist())

    return (
        np.asarray(targets_all),
        np.asarray(predictions_all),
    )


@torch.inference_mode()
def latency(model, device, warmup=50, runs=300):

    x = torch.randn(
        1, 3, 32, 32,
        device=device
    )

    model.eval()

    for _ in range(warmup):
        model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []

    for _ in range(runs):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        times_ms.append(
            (end - start) * 1000
        )

    return {
        "latency_mean_ms": float(
            statistics.mean(times_ms)
        ),

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
            1000 / statistics.mean(times_ms)
        ),
    }


def model_size_mb(model_state_dict):

    total_bytes = 0

    for tensor in model_state_dict.values():
        total_bytes += (
            tensor.numel()
            *
            tensor.element_size()
        )

    return total_bytes / (1024 ** 2)



def load_checkpoint(path):

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )



def main(args):

    set_seed(args.seed)

    device = get_device(args.device)

    checkpoint_path = Path(args.checkpoint)

    checkpoint = load_checkpoint(
        checkpoint_path
    )


    checkpoint_config = checkpoint.get(
        "config",
        {}
    )


    model_name = args.model or checkpoint_config.get(
        "model",
        "tinycnn"
    )

    model_name = canonical_model_name(
        model_name
    )


    model = build_model(model_name)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()


    test_loader = build_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


    y_true, y_pred = predict(
        model,
        test_loader,
        device
    )


    # ==========================
    # Classification Metrics
    # ==========================

    metrics = {

        "model": model_name,

        "precision_type": "FP32",

        "checkpoint_epoch":
            int(checkpoint.get("epoch", -1)),


        "best_validation_accuracy":
            float(
                checkpoint.get(
                    "best_val_acc",
                    -1
                )
            ),


        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),


        "macro_precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            ),


        "macro_recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            ),


        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro"
                )
            ),


        "trainable_parameters":
            count_parameters(model),


        "model_size_mb":
            model_size_mb(
                checkpoint["model_state_dict"]
            ),


        "evaluation_device":
            str(device),

    }



    # ==========================
    # Detailed Report
    # ==========================

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )


    cm = confusion_matrix(
        y_true,
        y_pred
    )



    save_json(
        report,
        checkpoint_path.parent /
        "classification_report.json"
    )


    save_json(
        {
            "confusion_matrix":
                cm.tolist()
        },

        checkpoint_path.parent /
        "confusion_matrix.json"
    )



    # ==========================
    # Latency
    # ==========================

    if args.measure_latency:

        latency_device = get_device(
            args.latency_device
        )


        latency_model = build_model(
            model_name
        )


        latency_model.load_state_dict(
            checkpoint["model_state_dict"]
        )


        latency_model.to(
            latency_device
        )

        latency_model.eval()


        if latency_device.type == "cpu":

            torch.set_num_threads(
                args.cpu_threads
            )


            torch.set_num_interop_threads(
                1
            )


        metrics["latency_device"] = str(
            latency_device
        )


        metrics.update(
            latency(
                latency_model,
                latency_device,
                warmup=args.warmup,
                runs=args.runs
            )
        )


    # ==========================
    # Save
    # ==========================

    output_path = (
        checkpoint_path.parent /
        "metrics.json"
    )


    save_json(
        metrics,
        output_path
    )



    print("\n")
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)


    for key, value in metrics.items():

        print(
            f"{key:30}: {value}"
        )


    print("\nSaved:")
    print(output_path.resolve())




if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--checkpoint",
        required=True
    )


    parser.add_argument(
        "--model",
        default=None
    )


    parser.add_argument(
        "--data-dir",
        default="./data"
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=256
    )


    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )


    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cuda",
            "cpu"
        ]
    )


    parser.add_argument(
        "--measure-latency",
        action="store_true"
    )


    parser.add_argument(
        "--latency-device",
        default="cpu",
        choices=[
            "auto",
            "cuda",
            "cpu"
        ]
    )


    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1
    )


    parser.add_argument(
        "--warmup",
        type=int,
        default=50
    )


    parser.add_argument(
        "--runs",
        type=int,
        default=300
    )


    main(
        parser.parse_args()
    )