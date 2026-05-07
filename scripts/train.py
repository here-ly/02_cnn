import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision")

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.transforms import build_datasets, build_loaders
from src.models.backbone import build_model
from src.losses.registry import build_loss
from src.metrics.collector import MetricCollector
from src.engine.trainer import Trainer


def main(config_path: str, resume_from: str = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config_dir = Path(config_path).parent

    wandb_config_path = config_dir / "wandb.yaml"
    use_wandb = wandb_config_path.exists()
    if use_wandb:
        try:
            import wandb
            with open(wandb_config_path, "r", encoding="utf-8") as f:
                wandb_cfg = yaml.safe_load(f) or {}
            # 允许主 config 的 wandb 字段覆盖 wandb.yaml（用于 kaggle 等不同环境）
            if "wandb" in config:
                wandb_cfg.update(config["wandb"])
            wandb_mode = wandb_cfg.get("mode", "offline")
            wandb.init(
                project=wandb_cfg.get("project", "cifar10-cnn"),
                entity=wandb_cfg.get("entity"),
                config=config,
                mode=wandb_mode,
                tags=wandb_cfg.get("tags", []),
                notes=wandb_cfg.get("notes", ""),
            )
            wandb.define_metric("epoch")
            wandb.define_metric("train/acc", step_metric="epoch")
            wandb.define_metric("test/acc", step_metric="epoch")
            wandb.define_metric("lr", step_metric="epoch")
            wandb.define_metric("timing/epoch_s", step_metric="epoch")
            wandb.define_metric("timing/forward_s", step_metric="epoch")
            wandb.define_metric("timing/backward_s", step_metric="epoch")
            wandb.define_metric("timing/data_s", step_metric="epoch")
        except (ImportError, Exception):
            use_wandb = False

    seed = config.get("seed", 42)
    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    data_cfg = config.get("data", {})
    train_dataset, test_dataset = build_datasets(
        data_root="./data",
        use_augmentation=data_cfg.get("use_augmentation", True),
    )

    test_mode = data_cfg.get("test_mode", False)
    test_cfg = data_cfg.get("test", {})
    train_loader, test_loader = build_loaders(
        train_dataset,
        test_dataset,
        batch_size=data_cfg.get("batch_size", 64),
        test_mode=test_mode,
        test_train_size=test_cfg.get("train_subset_size", 10000),
        test_test_size=test_cfg.get("test_subset_size", 2000),
        num_workers=data_cfg.get("num_workers", 0),
    )

    model = build_model(config)
    model = model.to(device)

    criterion = build_loss()

    optim_cfg = config.get("optim", {})
    initial_lr = optim_cfg.get("lr", 0.001) * optim_cfg.get("lr_multiplier", 5)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        eps=optim_cfg.get("eps", 1e-8),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
    )

    collector = MetricCollector(use_wandb=use_wandb)

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        collector=collector,
    )

    start_epoch = 0
    if resume_from:
        start_epoch = trainer.load_state(resume_from)

    trainer.train(test_dataset=test_dataset, start_epoch=start_epoch)

    output_cfg = config.get("output", {})
    ckpt_dir = Path(output_cfg.get("checkpoint_dir", "./outputs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    print(f"Model saved to {ckpt_dir / 'final.pt'}")

    if use_wandb:
        wandb.finish()
        if wandb_mode == "offline":
            run_dir = wandb.run.dir if wandb.run else "wandb/"
            print(f"[wandb offline] 日志已保存到 {run_dir}")
            print("  上传命令: wandb sync wandb/offline-run-*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()
    main(args.config, resume_from=args.resume)
