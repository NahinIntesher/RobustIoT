import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

import argparse
import random
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
import torch.ao.quantization as tq

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from src.model import build_model, canonical_model_name
from src.utils import save_json, set_seed


MEAN=torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1)
STD=torch.tensor([0.2470,0.2435,0.2616]).view(3,1,1)


def unnormalize(x):
    return x*STD+MEAN


def normalize(x):
    return (x-MEAN)/STD


def clamp_image(x):
    return torch.clamp(x,0.0,1.0)


def gaussian_noise(x):
    return clamp_image(x+torch.randn_like(x)*0.08)


def shot_noise(x):
    scale=30.0
    return clamp_image(torch.poisson(x*scale)/scale)


def brightness(x):
    factor=random.uniform(0.7,1.3)
    return clamp_image(x*factor)


def contrast(x):
    factor=random.uniform(0.6,1.4)
    mean=x.mean(dim=(1,2),keepdim=True)
    return clamp_image((x-mean)*factor+mean)


def motion_blur(x):
    kernel=random.choice([3,5])
    padding=kernel//2
    weight=torch.ones(3,1,kernel,1)/kernel
    return clamp_image(
        F.conv2d(x.unsqueeze(0),weight,padding=(padding,0),groups=3).squeeze(0)
    )


def defocus_blur(x):
    kernel=random.choice([3,5])
    padding=kernel//2
    weight=torch.ones(3,1,kernel,kernel)/(kernel*kernel)
    return clamp_image(
        F.conv2d(x.unsqueeze(0),weight,padding=padding,groups=3).squeeze(0)
    )


def jpeg_compression(x):
    arr=(clamp_image(x).permute(1,2,0).numpy()*255).astype(np.uint8)
    image=Image.fromarray(arr)
    quality=random.choice([30,50,70])
    buffer=BytesIO()
    image.save(buffer,format="JPEG",quality=quality)
    buffer.seek(0)
    image=Image.open(buffer).convert("RGB")
    arr=np.asarray(image).astype(np.float32)/255.0
    return torch.from_numpy(arr).permute(2,0,1)


CORRUPTIONS=[
    gaussian_noise,
    shot_noise,
    motion_blur,
    defocus_blur,
    brightness,
    contrast,
    jpeg_compression,
]


def degrade_image(x):
    corruption=random.choice(CORRUPTIONS)
    return clamp_image(corruption(x))


class RPTQCalibrationDataset(Dataset):
    def __init__(self,base_dataset,degraded_ratio,seed):
        self.base_dataset=base_dataset
        self.seed=seed
        total=len(base_dataset)
        degraded_count=round(total*degraded_ratio)
        generator=np.random.default_rng(seed)
        degraded_indices=generator.choice(
            total,
            size=degraded_count,
            replace=False,
        )
        self.degraded_indices=set(degraded_indices.tolist())

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self,index):
        image,target=self.base_dataset[index]
        image=unnormalize(image)

        rng_state=random.getstate()
        np_state=np.random.get_state()
        torch_state=torch.random.get_rng_state()

        local_seed=self.seed+index
        random.seed(local_seed)
        np.random.seed(local_seed%(2**32))
        torch.manual_seed(local_seed)

        if index in self.degraded_indices:
            image=degrade_image(image)

        random.setstate(rng_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

        return normalize(clamp_image(image)),target


def build_calibration_loader(
    data_dir,
    batch_size,
    num_workers,
    degraded_ratio,
    seed,
):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914,0.4822,0.4465),
            (0.2470,0.2435,0.2616),
        ),
    ])

    train_set=datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    calibration_set=RPTQCalibrationDataset(
        train_set,
        degraded_ratio,
        seed,
    )

    return DataLoader(
        calibration_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


@torch.inference_mode()
def calibrate(model,loader):
    model.eval()

    print("\nRPTQ calibration started...")
    print(f"Calibration images : {len(loader.dataset)}")
    print(f"Calibration batches: {len(loader)}")

    for images,_ in tqdm(
        loader,
        total=len(loader),
        desc="Calibration",
        unit="batch",
    ):
        model(images)

    print("Calibration completed.")


def main(args):
    set_seed(args.seed)

    torch.backends.quantized.engine="x86"

    model_name=canonical_model_name(args.model)
    degraded_ratio=args.degraded_ratio/100.0

    print("="*65)
    print("RobustQuant-IoT | RPTQ Quantization")
    print("="*65)
    print(f"Model            : {model_name}")
    print(f"Degraded ratio   : {args.degraded_ratio}%")
    print("Calibration data : CIFAR-10 training set")
    print("Calibration size : 45,000 images")
    print("Test set         : NOT used")
    print("Backend          : x86")
    print(f"Seed             : {args.seed}")
    print("="*65)

    checkpoint=torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    state_dict=checkpoint.get("model_state_dict")

    if state_dict is None:
        state_dict=checkpoint.get("state_dict")

    if state_dict is None:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict' or 'state_dict'."
        )

    model=build_model(model_name)
    model.load_state_dict(state_dict,strict=True)
    model.eval()

    print("\nLoaded FP32 checkpoint.")

    example_inputs=(torch.randn(1,3,32,32),)

    qconfig_dict={"":tq.get_default_qconfig("x86")}

    print("Preparing FX quantization graph...")

    model=prepare_fx(
        model,
        qconfig_dict,
        example_inputs,
    )

    calibration_loader=build_calibration_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        degraded_ratio=degraded_ratio,
        seed=args.seed,
    )

    calibrate(model,calibration_loader)

    print("\nConverting calibrated model -> INT8 RPTQ...")

    model=convert_fx(model)
    model.eval()

    output_dir=Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    model_filename=f"{model_name}_rptq_{args.degraded_ratio}pct_int8_model.pth"
    model_path=output_dir/model_filename

    torch.save(
        {
            "model_name":model_name,
            "state_dict":model.state_dict(),
            "precision":"INT8_RPTQ",
            "degraded_ratio":args.degraded_ratio,
            "calibration_images":len(calibration_loader.dataset),
            "calibration_source":"CIFAR10_train",
            "backend":"x86",
            "seed":args.seed,
        },
        model_path,
    )

    config={
        "model":model_name,
        "precision":"INT8_RPTQ",
        "degraded_ratio_percent":args.degraded_ratio,
        "calibration_images":len(calibration_loader.dataset),
        "degraded_images":round(
            len(calibration_loader.dataset)*degraded_ratio
        ),
        "clean_images":len(calibration_loader.dataset)-round(
            len(calibration_loader.dataset)*degraded_ratio
        ),
        "calibration_source":"CIFAR10_train",
        "test_set_used_for_calibration":False,
        "backend":"x86",
        "seed":args.seed,
    }

    save_json(
        config,
        output_dir/"config.json",
    )

    print("\n"+"="*65)
    print("RPTQ COMPLETE")
    print("="*65)
    print(f"INT8 model saved:")
    print(model_path.resolve())
    print("\nCalibration summary:")
    print(f"Total    : {config['calibration_images']}")
    print(f"Clean    : {config['clean_images']}")
    print(f"Degraded : {config['degraded_images']}")
    print("="*65)


if __name__=="__main__":
    parser=argparse.ArgumentParser(
        description="Robust Post-Training Quantization"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["tinycnn","mobilenetv2"],
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--degraded-ratio",
        type=int,
        required=True,
        choices=[25,50,75],
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    main(parser.parse_args())