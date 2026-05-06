import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics.collector import compute_confusion_probability


def evaluate_model(model: nn.Module, device: torch.device, test_loader: DataLoader) -> dict:
    model.eval()
    correct = 0
    total = 0
    confusion_counts = np.zeros((10, 10), dtype=np.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true = labels.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            for t, p in zip(y_true, y_pred):
                confusion_counts[t, p] += 1

    accuracy = 100.0 * correct / max(total, 1)
    confusion_prob = compute_confusion_probability(confusion_counts)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "confusion_counts": confusion_counts,
        "confusion_prob": confusion_prob,
    }
