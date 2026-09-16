# evaluate_int8.py

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.dataset import build_test_loader
from src.model import build_model, canonical_model_name
from src.utils import save_json, set_seed



# =====================================================
# Same QuantWrapper used in quantize_ptq.py
# =====================================================

class QuantWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()


    def forward(self, x):

        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)

        return x



# =====================================================
# Prepare identical INT8 architecture
# =====================================================

def build_int8_model(model_name):

    torch.backends.quantized.engine = "x86"


    fp32_model = build_model(
        model_name
    )


    model = QuantWrapper(
        fp32_model
    )


    model.eval()


    model.qconfig = tq.get_default_qconfig(
        "x86"
    )


    model = tq.prepare(
        model
    )


    model = tq.convert(
        model
    )


    return model



# =====================================================
# Evaluation
# =====================================================

@torch.inference_mode()
def predict(model, loader):

    y_true = []
    y_pred = []


    model.eval()


    for images, targets in loader:

        outputs = model(images)

        preds = outputs.argmax(
            dim=1
        )


        y_true.extend(
            targets.tolist()
        )

        y_pred.extend(
            preds.tolist()
        )


    return (
        np.array(y_true),
        np.array(y_pred)
    )



# =====================================================
# Latency
# =====================================================

@torch.inference_mode()
def measure_latency(model, warmup=50, runs=300):

    model.eval()


    x = torch.randn(
        1,3,32,32
    )


    for _ in range(warmup):

        model(x)



    times=[]


    for _ in range(runs):

        start=time.perf_counter()

        model(x)

        end=time.perf_counter()


        times.append(
            (end-start)*1000
        )


    return {

        "latency_mean_ms":
            float(statistics.mean(times)),

        "latency_median_ms":
            float(statistics.median(times)),

        "latency_p95_ms":
            float(np.percentile(times,95)),

        "latency_p99_ms":
            float(np.percentile(times,99)),

        "throughput_inf_per_s":
            float(
                1000/statistics.mean(times)
            )
    }



# =====================================================
# Main
# =====================================================

def main(args):

    set_seed(
        args.seed
    )


    checkpoint_path = Path(
        args.checkpoint
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )


    model_name = canonical_model_name(
        checkpoint["model_name"]
    )


    print("="*60)
    print("INT8 PTQ Evaluation")
    print("="*60)


    print(
        "Model:",
        model_name
    )


    print(
        "Device: CPU"
    )


    # Recreate INT8 model

    model = build_int8_model(
        model_name
    )


    # Load quantized weights

    model.load_state_dict(
        checkpoint["state_dict"]
    )


    model.eval()



    test_loader = build_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


    y_true, y_pred = predict(
        model,
        test_loader
    )



    metrics = {

        "precision_type":
            "INT8_PTQ",

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

        "model_size_mb":
            checkpoint_path.stat().st_size /
            (1024**2)

    }


    metrics.update(
        measure_latency(
            model,
            args.warmup,
            args.runs
        )
    )



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


    output_dir = checkpoint_path.parent


    save_json(
        metrics,
        output_dir /
        "int8_metrics.json"
    )


    save_json(
        report,
        output_dir /
        "int8_classification_report.json"
    )


    save_json(
        {
            "confusion_matrix":
                cm.tolist()
        },
        output_dir /
        "int8_confusion_matrix.json"
    )


    print("\nRESULTS")
    print("="*60)


    for k,v in metrics.items():

        print(
            f"{k}: {v}"
        )


    print("\nSaved:")
    print(
        output_dir.resolve()
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--checkpoint",
        required=True
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