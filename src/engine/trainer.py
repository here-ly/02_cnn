import gc
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from src.utils.timer import Timer
from src.metrics.collector import (
    compute_confusion_probability,
    save_metrics_figure,
    save_gradient_heatmaps,
)


def build_scheduler(optimizer, config: dict):
    sched_cfg = config.get("train", {}).get("scheduler", {})
    sched_type = sched_cfg.get("type", "ReduceLROnPlateau")

    if sched_type == "CosineAnnealing":
        epochs = config.get("train", {}).get("epochs", 100)
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=sched_cfg.get("min_lr", 1e-6),
        )

    factor = sched_cfg.get("factor", 0.5)
    patience = sched_cfg.get("patience", 5)
    min_lr = sched_cfg.get("min_lr", 1e-6)
    return ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


class Trainer:
    def __init__(self, model, device, train_loader, test_loader, criterion, optimizer, config, collector=None):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.collector = collector

        self.train_cfg = config.get("train", {})
        self.epochs = self.train_cfg.get("epochs", 100)
        self.log_interval = self.train_cfg.get("log_interval", 100)
        self.save_interval = self.train_cfg.get("save_interval", 10)
        self.save_heatmaps = self.train_cfg.get("save_heatmaps", True)
        self.scheduler_cfg = self.train_cfg.get("scheduler", {})

        self.output_cfg = config.get("output", {})
        self.image_dir = self.output_cfg.get("image_dir", "./information_image")
        self.checkpoint_dir = Path(self.output_cfg.get("checkpoint_dir", "./outputs/checkpoints"))

        self.train_acc_history = []
        self.test_acc_history = []
        self.best_test_acc = 0.0
        self.current_lr = optimizer.param_groups[0]["lr"]
        self.global_step = 0

        self.scheduler = build_scheduler(optimizer, config)
        self._use_plateau = isinstance(self.scheduler, ReduceLROnPlateau)

    def _log(self, metrics: dict, step: int):
        if self.collector is None:
            return
        try:
            self.collector.log(metrics, step=step)
        except Exception as e:
            print(f"[wandb log error, training continues] {e}")

    def load_state(self, ckpt_path: str) -> int:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.train_acc_history = ckpt.get("train_acc_history", [])
        self.test_acc_history = ckpt.get("test_acc_history", [])
        self.best_test_acc = max(self.test_acc_history) if self.test_acc_history else 0.0
        self.global_step = ckpt.get("global_step", 0)
        self.current_lr = self.optimizer.param_groups[0]["lr"]
        resume_epoch = ckpt["epoch"]  # 已完成的 epoch 数，下一轮从此开始
        print(f"[resume] 从 epoch {resume_epoch + 1} 继续训练 "
              f"(best test acc={self.best_test_acc:.1f}%, global_step={self.global_step})")
        return resume_epoch

    def _save_checkpoint(self, tag: str):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{tag}.pt"
        torch.save({
            "epoch": len(self.test_acc_history),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
            "train_acc_history": self.train_acc_history,
            "test_acc_history": self.test_acc_history,
        }, path)
        print(f"[checkpoint saved] {path}")

    def train_epoch(self, epoch: int):
        self.model.train()
        train_correct = 0
        train_total = 0
        epoch_start = time.perf_counter()
        cum_data = 0.0
        cum_forward = 0.0
        cum_backward = 0.0

        for i, (images, labels) in enumerate(self.train_loader):
            with Timer() as t_data:
                images = images.to(self.device)
                labels = labels.to(self.device)

            with Timer() as t_forward:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            with Timer() as t_backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            cum_data += t_data.elapsed
            cum_forward += t_forward.elapsed
            cum_backward += t_backward.elapsed

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[NaN/Inf loss detected at epoch {epoch+1}, step {i+1}]")
                self._save_checkpoint(f"crash_epoch{epoch+1}_step{i+1}")
                raise RuntimeError(f"Loss is NaN/Inf at epoch {epoch+1}, step {i+1}. Checkpoint saved.")

            batch_acc = (outputs.argmax(1) == labels).float().mean()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            self.global_step += 1

            if (i + 1) % self.log_interval == 0:
                self._log({
                    "step/loss": loss.item(),
                    "step/acc": batch_acc.item() * 100,
                    "step/lr": self.current_lr,
                }, step=self.global_step)

        epoch_elapsed = time.perf_counter() - epoch_start
        train_acc = 100.0 * train_correct / max(train_total, 1)
        self.train_acc_history.append(train_acc)

        timing = {
            "timing/epoch_s": epoch_elapsed,
            "timing/data_s": cum_data,
            "timing/forward_s": cum_forward,
            "timing/backward_s": cum_backward,
        }
        return train_acc, timing

    def evaluate(self, epoch: int, test_dataset=None):
        self.model.eval()
        correct = 0
        total = 0
        confusion_counts = np.zeros((10, 10), dtype=np.int64)

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_true = labels.cpu().numpy()
                y_pred = predicted.cpu().numpy()
                for t, p in zip(y_true, y_pred):
                    confusion_counts[t, p] += 1

        test_acc = 100.0 * correct / max(total, 1)
        self.test_acc_history.append(test_acc)

        if self._use_plateau:
            self.scheduler.step(test_acc)
        else:
            self.scheduler.step()
        self.current_lr = self.optimizer.param_groups[0]["lr"]

        confusion_prob = compute_confusion_probability(confusion_counts)
        try:
            save_metrics_figure(epoch, self.train_acc_history, confusion_prob, self.image_dir)
        except Exception as e:
            print(f"[metrics figure save error] {e}")

        self._log({
            "epoch": epoch + 1,
            "train/acc": self.train_acc_history[-1],
            "test/acc": test_acc,
            "lr": self.current_lr,
        }, step=self.global_step)

        if self.save_heatmaps and test_dataset is not None:
            try:
                save_gradient_heatmaps(self.model, self.device, test_dataset, self.image_dir, epoch, sample_indices=[0, 1])
            except Exception as e:
                print(f"[heatmap save error] {e}")

        gc.collect()
        return test_acc

    def train(self, test_dataset=None, start_epoch: int = 0):
        for epoch in range(start_epoch, self.epochs):
            _, timing = self.train_epoch(epoch)
            test_acc = self.evaluate(epoch, test_dataset=test_dataset)
            timing["epoch"] = epoch + 1
            self._log(timing, step=self.global_step)

            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self._save_checkpoint("best")

            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f"epoch_{epoch+1:03d}")
