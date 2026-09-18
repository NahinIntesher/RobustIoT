import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from tqdm import tqdm

from src.dataset import build_loaders
from src.model import build_model
from src.utils import set_seed


def load_fp32_checkpoint(path, model_name):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    saved_model_name = checkpoint.get(
        "model_name",
        checkpoint.get(
            "config",
            {}
        ).get("model", model_name),
    )

    return checkpoint, state_dict, saved_model_name


def prepare_qat_model(model_name, state_dict):
    torch.backends.quantized.engine = "x86"

    model = build_model(
        model_name,
        num_classes=10,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.train()

    qconfig_mapping = (
        tq.QConfigMapping()
        .set_global(
            tq.get_default_qat_qconfig("x86")
        )
    )

    example_inputs = (
        torch.randn(1, 3, 32, 32),
    )

    model = prepare_qat_fx(
        model,
        qconfig_mapping,
        example_inputs,
    )

    model.train()

    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Train",
        unit="batch",
        leave=False,
    )

    for images, targets in progress:
        images = images.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)
        loss = criterion(
            outputs,
            targets,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        correct += (
            outputs.argmax(dim=1) == targets
        ).sum().item()

        total += targets.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.2f}%",
        )

    return (
        total_loss / total,
        correct / total,
    )


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Val  ",
        unit="batch",
        leave=False,
    )

    for images, targets in progress:
        images = images.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        correct += (
            outputs.argmax(dim=1) == targets
        ).sum().item()

        total += targets.size(0)

        progress.set_postfix(
            acc=f"{100.0 * correct / total:.2f}%"
        )

    return correct / total


def freeze_bn_stats(model):
    for module in model.modules():
        if isinstance(
            module,
            nn.modules.batchnorm._BatchNorm,
        ):
            module.eval()


def freeze_observers(model):
    for module in model.modules():
        if hasattr(
            module,
            "disable_observer",
        ):
            module.disable_observer()


def save_qat_checkpoint(
    path,
    model,
    model_name,
    best_val,
    seed,
    epochs,
):
    torch.save(
        {
            "model_name": model_name,
            "precision_type": "QAT",
            "qat_state_dict": copy.deepcopy(
                model.state_dict()
            ),
            "best_val_accuracy": float(best_val),
            "seed": int(seed),
            "epochs": int(epochs),
            "backend": "x86",
        },
        path,
    )


def main(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 65)
    print("RobustQuant-IoT | QAT Training")
    print("=" * 65)
    print(f"Model  : {args.model}")
    print(f"Device : {device}")
    print(f"Epochs : {args.epochs}")
    print(f"Seed   : {args.seed}")

    checkpoint, state_dict, model_name = (
        load_fp32_checkpoint(
            args.checkpoint,
            args.model,
        )
    )

    print("Loaded FP32 checkpoint")

    model = prepare_qat_model(
        model_name,
        state_dict,
    )

    model = model.to(device)

    train_loader, val_loader, _ = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=42,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_val = -1.0
    best_state = None

    print("\nQAT training started...\n")

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()

        if epoch == args.freeze_bn_epoch:
            freeze_bn_stats(model)
            print(
                f"Epoch {epoch}: BatchNorm statistics frozen."
            )

        if epoch == args.freeze_observer_epoch:
            freeze_observers(model)
            print(
                f"Epoch {epoch}: Quantization observers frozen."
            )

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_acc = evaluate(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        elapsed = (
            time.perf_counter() - start
        )

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"| Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc * 100:.2f}% "
            f"| Val Acc: {val_acc * 100:.2f}% "
            f"| Time: {elapsed:.1f}s"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(
                model.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "No valid QAT checkpoint was produced."
        )

    print(
        f"\nBest validation accuracy: "
        f"{best_val * 100:.2f}%"
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the best fake-quantized QAT model.
    qat_path = (
        output_dir / "qat_model.pth"
    )

    model.load_state_dict(
        best_state
    )

    save_qat_checkpoint(
        qat_path,
        model,
        model_name,
        best_val,
        args.seed,
        args.epochs,
    )

    print(
        f"Saved QAT checkpoint:\n"
        f"{qat_path.resolve()}"
    )

    # Convert the best QAT model to real INT8.
    print("\nConverting QAT model -> INT8...")

    model.eval()
    model = model.to("cpu")

    int8_model = convert_fx(model)

    int8_path = (
        output_dir
        / "qat_int8_model.pth"
    )

    torch.save(
        {
            "model_name": model_name,
            "precision_type": "INT8_QAT",
            "state_dict": int8_model.state_dict(),
            "best_val_accuracy": float(
                best_val
            ),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "backend": "x86",
            "source_qat_checkpoint": "qat_model.pth",
        },
        int8_path,
    )

    print()
    print("=" * 65)
    print("DONE")
    print("=" * 65)
    print(
        f"QAT checkpoint:\n{qat_path.resolve()}"
    )
    print(
        f"\nINT8 checkpoint:\n{int8_path.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, choices=["tinycnn", "mobilenetv2"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-bn-epoch", type=int, default=35)
    parser.add_argument("--freeze-observer-epoch", type=int, default=40)

    main(parser.parse_args())