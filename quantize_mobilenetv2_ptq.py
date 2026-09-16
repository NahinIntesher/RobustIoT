# quantize_mobilenetv2_ptq.py

import argparse
from pathlib import Path

import torch
import torch.ao.quantization as tq

from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    convert_fx
)

from src.dataset import build_test_loader
from src.model import build_model, canonical_model_name
from src.utils import set_seed, save_json


@torch.inference_mode()
def calibrate(model, loader, batches):

    model.eval()

    print("\nCalibration started...")

    count = 0

    for images, _ in loader:

        model(images)

        count += 1

        if count >= batches:
            break


    print(
        f"Calibration completed: {count} batches"
    )



def main(args):

    set_seed(args.seed)

    torch.backends.quantized.engine = "x86"


    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )


    model_name = canonical_model_name(
        args.model
    )


    print("="*60)
    print("MobileNetV2 INT8 PTQ")
    print("="*60)

    print(
        "Model:",
        model_name
    )


    model = build_model(
        model_name
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.eval()



    example_inputs = (
        torch.randn(
            1,
            3,
            32,
            32
        ),
    )


    qconfig = tq.get_default_qconfig(
        "x86"
    )


    qconfig_dict = {
        "": qconfig
    }



    print(
        "\nPreparing FX graph..."
    )


    model = prepare_fx(
        model,
        qconfig_dict,
        example_inputs
    )



    loader = build_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


    calibrate(
        model,
        loader,
        args.calibration_batches
    )



    print(
        "\nConverting INT8..."
    )


    model = convert_fx(
        model
    )


    model.eval()



    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    save_path = (
        output_dir /
        "mobilenetv2_int8_model.pth"
    )


    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "precision": "INT8_PTQ"
        },
        save_path
    )



    save_json(
        {
            "model": model_name,
            "precision": "INT8_PTQ",
            "backend": "x86"
        },
        output_dir /
        "config.json"
    )


    print("\nDONE")
    print(
        "Saved:",
        save_path.resolve()
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--checkpoint",
        required=True
    )


    parser.add_argument(
        "--model",
        default="mobilenetv2"
    )


    parser.add_argument(
        "--data-dir",
        default="./data"
    )


    parser.add_argument(
        "--output-dir",
        default="experiments/mobilenetv2_ptq/seed_42"
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )


    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )


    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=100
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )


    main(
        parser.parse_args()
    )