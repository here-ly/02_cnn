import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.transforms import denormalize_image


class MetricCollector:
    def __init__(self, use_wandb: bool = False):
        self.use_wandb = use_wandb
        self._wandb = None
        self._fail_count = 0
        if self.use_wandb:
            try:
                import wandb as wb
                self._wandb = wb
            except ImportError:
                self.use_wandb = False

    def log(self, metrics: dict, step: int = None):
        if not self.use_wandb or self._wandb is None:
            return
        for attempt in range(3):
            try:
                self._wandb.log(metrics, step=step)
                self._fail_count = 0
                return
            except OSError:
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
        self._fail_count += 1
        if self._fail_count <= 3 or self._fail_count % 10 == 1:
            print(f"[wandb log error #{self._fail_count}, training continues]")

    def log_image(self, key: str, path: str, step: int = None):
        if self.use_wandb and self._wandb is not None:
            self._wandb.log({key: self._wandb.Image(path)}, step=step)


def compute_confusion_probability(confusion_counts: np.ndarray) -> np.ndarray:
    row_sum = confusion_counts.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return confusion_counts / row_sum


def save_metrics_figure(epoch: int, train_acc_history: list, test_confusion_prob: np.ndarray, output_dir: str):
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    xs = np.arange(1, len(train_acc_history) + 1)
    ax1.plot(xs, train_acc_history, marker="o", label="Train Acc")
    ax1.axhline(90.0, linestyle="--", color="r", label="Target 90%")
    ax1.set_title(f"Epoch {epoch + 1} Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(test_confusion_prob, cmap="YlOrRd", vmin=0, vmax=1)
    ax2.set_title(f"Epoch {epoch + 1} P(pred=b|true=a)")
    ax2.set_xlabel("Predicted label")
    ax2.set_ylabel("True label")
    ax2.set_xticks(np.arange(10))
    ax2.set_yticks(np.arange(10))
    for i in range(10):
        for j in range(10):
            color = "white" if test_confusion_prob[i, j] > 0.5 else "black"
            ax2.text(j, i, f"{test_confusion_prob[i, j]:.2f}", ha="center", va="center", color=color, fontsize=7)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"epoch_{epoch + 1:03d}_metrics.png")
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def save_gradient_heatmaps(model, device, dataset, output_dir: str, epoch: int, sample_indices=None):
    if sample_indices is None:
        sample_indices = [0, 1]
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    for draw_id, idx in enumerate(sample_indices, start=1):
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)
        image.requires_grad_(True)

        output = model(image)
        pred_class = output.argmax(dim=1).item()
        score = output[0, pred_class]
        model.zero_grad()
        score.backward()

        grad = image.grad.detach().abs()[0]
        heatmap = grad.max(dim=0)[0].cpu().numpy()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        image_vis = denormalize_image(image.detach())[0].permute(1, 2, 0).cpu().numpy()

        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(image_vis)
        ax1.set_title(f"idx={idx}  true={label}  pred={pred_class}")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(image_vis)
        ax2.imshow(heatmap, cmap="jet", alpha=0.5)
        ax2.set_title("Gradient Heatmap")
        ax2.axis("off")

        fig.tight_layout()
        save_path = os.path.join(output_dir, f"epoch_{epoch + 1:03d}_grad_heatmap_{draw_id}.png")
        fig.savefig(save_path, dpi=140)
        plt.close(fig)
