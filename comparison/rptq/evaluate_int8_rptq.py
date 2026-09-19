import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

import argparse
import statistics
import time

import numpy as np
import torch
import torch.ao.quantization as tq
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from tqdm import tqdm

from src.dataset import build_test_loader
from src.model import build_model
from src.utils import save_json, set_seed


def rebuild_rptq_model(model_name,state_dict):
    torch.backends.quantized.engine="x86"

    model=build_model(model_name)
    model.eval()

    example_inputs=(torch.randn(1,3,32,32),)
    qconfig_dict={"":tq.get_default_qconfig("x86")}

    model=prepare_fx(
        model,
        qconfig_dict,
        example_inputs,
    )

    model=convert_fx(model)

    model.load_state_dict(state_dict,strict=True)
    model.eval()

    return model


@torch.inference_mode()
def evaluate(model,loader):
    y_true=[]
    y_pred=[]

    print("\nEvaluating clean CIFAR-10...")

    for images,targets in tqdm(
        loader,
        total=len(loader),
        desc="Evaluation",
        unit="batch",
    ):
        outputs=model(images)
        preds=outputs.argmax(dim=1)

        y_true.extend(targets.tolist())
        y_pred.extend(preds.tolist())

    return np.asarray(y_true),np.asarray(y_pred)


@torch.inference_mode()
def measure_latency(model,warmup=50,runs=300):
    x=torch.randn(1,3,32,32)

    for _ in range(warmup):
        model(x)

    times=[]

    for _ in range(runs):
        start=time.perf_counter()
        model(x)
        times.append((time.perf_counter()-start)*1000)

    mean=statistics.mean(times)

    return {
        "latency_mean_ms":float(mean),
        "latency_median_ms":float(statistics.median(times)),
        "latency_p95_ms":float(np.percentile(times,95)),
        "latency_p99_ms":float(np.percentile(times,99)),
        "throughput_inf_per_s":float(1000/mean),
    }


def main(args):
    set_seed(args.seed)

    torch.backends.quantized.engine="x86"

    checkpoint_path=Path(args.checkpoint)

    print("="*65)
    print("RobustQuant-IoT | RPTQ Evaluation")
    print("="*65)
    print(f"Checkpoint : {checkpoint_path}")
    print("Dataset    : CIFAR-10 Test")
    print("Device     : CPU")
    print("="*65)

    checkpoint=torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_name=checkpoint["model_name"]
    state_dict=checkpoint["state_dict"]

    print(f"Model      : {model_name}")
    print(
        f"Degraded calibration : "
        f"{checkpoint.get('degraded_ratio', 'unknown')}%"
    )

    model=rebuild_rptq_model(
        model_name,
        state_dict,
    )

    test_loader=build_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    y_true,y_pred=evaluate(
        model,
        test_loader,
    )

    metrics={
        "precision_type":"INT8_RPTQ",
        "model":model_name,
        "degraded_calibration_percent":checkpoint.get(
            "degraded_ratio",
            None,
        ),
        "accuracy":float(
            accuracy_score(y_true,y_pred)
        ),
        "macro_precision":float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall":float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1":float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "model_size_mb":float(
            checkpoint_path.stat().st_size/(1024**2)
        ),
    }

    metrics.update(
        measure_latency(
            model,
            args.warmup,
            args.runs,
        )
    )

    output_dir=checkpoint_path.parent

    save_json(
        metrics,
        output_dir/"rptq_clean_metrics.json",
    )

    save_json(
        classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        output_dir/"classification_report.json",
    )

    save_json(
        {
            "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred,
            ).tolist()
        },
        output_dir/"confusion_matrix.json",
    )

    print("\n"+"="*65)
    print("RESULTS")
    print("="*65)

    for key,value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(
        (output_dir/"rptq_clean_metrics.json").resolve()
    )
    print(
        (output_dir/"classification_report.json").resolve()
    )
    print(
        (output_dir/"confusion_matrix.json").resolve()
    )


if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=300,
    )

    main(parser.parse_args())