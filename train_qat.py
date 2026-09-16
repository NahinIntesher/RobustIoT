# train_qat.py

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq

from torch.ao.quantization import get_default_qat_qconfig

from src.dataset import build_loaders
from src.model import (
    build_model,
    canonical_model_name,
    count_parameters,
)
from src.utils import (
    set_seed,
    save_json,
)



# ==========================================================
# QAT Preparation
# ==========================================================

def prepare_qat_model(model):

    torch.backends.quantized.engine = "x86"

    model.train()


    model.qconfig = get_default_qat_qconfig(
        "x86"
    )


    tq.prepare_qat(
        model,
        inplace=True
    )


    return model



# ==========================================================
# Train Epoch
# ==========================================================

def train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device):


    model.train()


    total_loss = 0
    correct = 0
    total = 0


    for images, targets in loader:

        images = images.to(device)
        targets = targets.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            targets
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()


        pred = outputs.argmax(
            dim=1
        )


        correct += (
            pred == targets
        ).sum().item()


        total += targets.size(0)



    return (
        total_loss / len(loader),
        100 * correct / total
    )



# ==========================================================
# Validation
# ==========================================================

@torch.no_grad()
def evaluate(
        model,
        loader,
        criterion,
        device):


    model.eval()


    loss_total = 0
    correct = 0
    total = 0


    for images, targets in loader:

        images = images.to(device)
        targets = targets.to(device)


        outputs = model(images)


        loss_total += criterion(
            outputs,
            targets
        ).item()


        pred = outputs.argmax(
            dim=1
        )


        correct += (
            pred == targets
        ).sum().item()


        total += targets.size(0)



    return (
        loss_total / len(loader),
        100 * correct / total
    )



# ==========================================================
# Main
# ==========================================================

def main(args):

    set_seed(
        args.seed
    )


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    print("="*65)
    print("RobustQuant-IoT | QAT Training")
    print("="*65)



    model_name = canonical_model_name(
        args.model
    )


    model = build_model(
        model_name
    )



    # Load FP32 pretrained model

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )



    print(
        "Loaded FP32 checkpoint"
    )


    print(
        "Model:",
        model_name
    )


    print(
        "Device:",
        device
    )



    # Prepare QAT

    model = prepare_qat_model(
        model
    )


    model.to(device)



    train_loader, val_loader, _ = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=42
    )



    criterion = nn.CrossEntropyLoss()



    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )



    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )



    best_acc = 0



    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )



    for epoch in range(1, args.epochs+1):

        start=time.time()


        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )


        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )


        scheduler.step()



        print("\n" + "="*60)
        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Acc : {train_acc:.2f}%"
        )

        print(
            f"Val Loss  : {val_loss:.4f}"
        )

        print(
            f"Val Acc   : {val_acc:.2f}%"
        )

        print(
            f"Time      : {time.time()-start:.1f}s"
        )



        if val_acc > best_acc:

            best_acc = val_acc


            torch.save(
                {
                    "model_name": model_name,
                    "epoch": epoch,
                    "best_val_acc": best_acc,
                    "model_state_dict":
                        copy.deepcopy(
                            model.state_dict()
                        )
                },
                output_dir /
                "qat_best_checkpoint.pth"
            )



    # Convert QAT model to INT8

    print("\nConverting QAT model to INT8...")


    model.cpu()
    model.eval()


    quantized_model = tq.convert(
        model,
        inplace=False
    )


    torch.save(
        {
            "model_name": model_name,
            "precision":
                "INT8_QAT",
            "state_dict":
                quantized_model.state_dict(),
            "best_val_acc":
                best_acc
        },
        output_dir /
        "qat_int8_model.pth"
    )


    save_json(
        {
            "model":
                model_name,
            "precision":
                "INT8_QAT",
            "best_val_acc":
                best_acc,
            "parameters":
                count_parameters(model)
        },
        output_dir /
        "qat_config.json"
    )


    print("\nDONE")
    print(
        "Best Val Accuracy:",
        best_acc
    )

    print(
        "Saved:",
        output_dir.resolve()
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model",
        default="tinycnn"
    )


    parser.add_argument(
        "--checkpoint",
        required=True
    )


    parser.add_argument(
        "--output-dir",
        required=True
    )


    parser.add_argument(
        "--data-dir",
        default="./data"
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )


    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
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