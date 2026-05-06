import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision")

from src.utils.device import get_device
from src.data.transforms import build_datasets, build_loaders
from src.models.backbone import build_model
from src.engine.evaluator import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    _, test_dataset = build_datasets(
        data_root="./data",
        use_augmentation=False,
    )

    data_cfg = config.get("data", {})
    _, test_loader = build_loaders(
        None, test_dataset,
        batch_size=data_cfg.get("batch_size", 64),
        test_mode=False,
        num_workers=data_cfg.get("num_workers", 0),
    )

    model = build_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)

    result = evaluate_model(model, device, test_loader)
    print(f"Test Accuracy: {result['accuracy']:.2f}%")
    print(f"Correct/Total: {result['correct']}/{result['total']}")
    print("Confusion Matrix (P(pred=b|true=a)):")
    np.set_printoptions(precision=3, suppress=True)
    print(result["confusion_prob"])


if __name__ == "__main__":
    main()
