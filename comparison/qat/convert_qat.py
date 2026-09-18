import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import torch
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

from src.model import build_model


def main(args):
    torch.backends.quantized.engine = "x86"

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_name = checkpoint["model_name"]
    qat_state_dict = checkpoint["qat_state_dict"]

    model = build_model(
        model_name,
        num_classes=10,
    )

    model.train()

    qconfig_mapping = tq.QConfigMapping().set_global(
        tq.get_default_qat_qconfig("x86")
    )

    example_inputs = (torch.randn(1, 3, 32, 32),)

    model = prepare_qat_fx(
        model,
        qconfig_mapping,
        example_inputs,
    )

    model.load_state_dict(
        qat_state_dict,
        strict=True,
    )

    model.eval()

    int8_model = convert_fx(model)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_name": model_name,
            "precision_type": "INT8_QAT",
            "state_dict": int8_model.state_dict(),
            "best_val_accuracy": checkpoint.get("best_val_accuracy"),
            "seed": checkpoint.get("seed"),
            "epochs": checkpoint.get("epochs"),
            "backend": "x86",
        },
        output_path,
    )

    print("=" * 60)
    print("QAT -> INT8 CONVERSION")
    print("=" * 60)
    print(f"Model      : {model_name}")
    print(f"Best Val   : {checkpoint.get('best_val_accuracy')}")
    print(f"Saved      : {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)

    main(parser.parse_args())