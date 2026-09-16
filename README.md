# RobustQuant-IoT — FP32 Baseline

Step 1 of the research implementation: reproducible CIFAR-10 + TinyCNN FP32 baseline.

CIFAR-10-C is intentionally NOT used for training. It is reserved for later corruption-robustness evaluation.

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
For TinyCNN FP32
-----------------------------
python train.py --model tinycnn --epochs 50 --seed 42 --split-seed 42
python train.py --model tinycnn --epochs 50 --seed 123 --split-seed 42
python train.py --model tinycnn --epochs 50 --seed 2026 --split-seed 42


For MobileNetV2 FP32
-----------------------------
python train.py --model mobilenetv2 --epochs 50 --seed 42 --split-seed 42
python train.py --model mobilenetv2 --epochs 50 --seed 123 --split-seed 42
python train.py --model mobilenetv2 --epochs 50 --seed 2026 --split-seed 42

python train.py --epochs 50 --batch-size 128 --lr 0.1 --seed 42
```


## Evaluate
```bash
For TinyCNN FP32
-----------------------------
python evaluate.py --checkpoint experiments/tinycnn_fp32/seed_42/best_model.pth
python evaluate.py --checkpoint experiments/tinycnn_fp32/seed_123/best_model.pth
python evaluate.py --checkpoint experiments/tinycnn_fp32/seed_2026/best_model.pth


For MobileNetV2 FP32
-----------------------------
python evaluate.py --checkpoint experiments/mobilenetv2_fp32/seed_42/best_model.pth
python evaluate.py --checkpoint experiments/mobilenetv2_fp32/seed_123/best_model.pth
python evaluate.py --checkpoint experiments/mobilenetv2_fp32/seed_2026/best_model.pth
```

Outputs are saved under `experiments/tinycnn_fp32/`: config, best checkpoint, CSV training log, plots, and clean-test metrics.

Keep the split, seed, preprocessing, and evaluation protocol fixed for later FP32/PTQ/QAT/RPTQ comparisons.
