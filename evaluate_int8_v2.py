# evaluate_int8_v2.py

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    convert_fx
)

import torch.ao.quantization as tq

from src.dataset import build_test_loader
from src.model import build_model
from src.utils import save_json, set_seed



def rebuild_int8_model(model_name, state_dict):

    torch.backends.quantized.engine = "x86"


    model = build_model(
        model_name
    )


    model.eval()


    example_inputs = (
        torch.randn(
            1,3,32,32
        ),
    )


    qconfig_dict = {
        "":
        tq.get_default_qconfig(
            "x86"
        )
    }


    model = prepare_fx(
        model,
        qconfig_dict,
        example_inputs
    )


    model = convert_fx(
        model
    )


    model.load_state_dict(
        state_dict
    )


    model.eval()


    return model



@torch.inference_mode()
def predict(model, loader):

    y_true=[]
    y_pred=[]


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


    return np.array(y_true), np.array(y_pred)



@torch.inference_mode()
def latency(model):

    x=torch.randn(
        1,3,32,32
    )


    for _ in range(50):
        model(x)


    times=[]


    for _ in range(300):

        start=time.perf_counter()

        model(x)

        end=time.perf_counter()

        times.append(
            (end-start)*1000
        )


    return {

        "latency_mean_ms":
            float(statistics.mean(times)),

        "throughput_inf_per_s":
            float(
                1000/statistics.mean(times)
            )
    }



def main(args):

    set_seed(args.seed)


    checkpoint=torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )


    model=rebuild_int8_model(
        checkpoint["model_name"],
        checkpoint["state_dict"]
    )


    loader=build_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


    y_true,y_pred=predict(
        model,
        loader
    )


    metrics={

        "precision_type":
            "INT8_PTQ_FX",

        "accuracy":
            float(
                accuracy_score(
                    y_true,y_pred
                )
            ),

        "macro_precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro"
                )
            ),

        "macro_recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro"
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
            Path(args.checkpoint).stat().st_size /
            (1024**2)

    }


    metrics.update(
        latency(model)
    )


    output_dir=Path(
        args.checkpoint
    ).parent


    save_json(
        metrics,
        output_dir /
        "mobilenetv2_int8_metrics.json"
    )


    save_json(
        classification_report(
            y_true,
            y_pred,
            output_dict=True
        ),
        output_dir /
        "classification_report.json"
    )


    save_json(
        {
            "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred
            ).tolist()
        },
        output_dir /
        "confusion_matrix.json"
    )


    print("="*60)
    print("RESULTS")
    print("="*60)


    for k,v in metrics.items():
        print(
            f"{k}: {v}"
        )



if __name__=="__main__":

    parser=argparse.ArgumentParser()


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


    main(
        parser.parse_args()
    )