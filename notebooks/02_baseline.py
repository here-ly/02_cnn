# ---
# jupytext:
#   formats: ipynb,py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
# ---

# %% [markdown]
# # 02 Baseline 模型训练与对比
#
# 本 notebook 用于：
# 1. 跑基线模型（无数据增强）观察过拟合
# 2. 跑数据增强模型对比泛化性能
# 3. 不同网络结构的对比实验
# 4. 学习率调整策略对比

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% [markdown]
# ## 1. 定义简单的 CNN baseline（无 BN, 无 dropout）

# %%
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

# %% [markdown]
# ## 2. 对比实验：无增强 vs 有增强

# %%
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

def get_loaders_with_aug():
    t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = torchvision.datasets.CIFAR10(root="../data" if Path.cwd().name == "notebooks" else "./data",
                                       train=True, download=True, transform=t)
    return DataLoader(ds, batch_size=64, shuffle=True)

def get_loader_no_aug(train=True):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = torchvision.datasets.CIFAR10(root="../data" if Path.cwd().name == "notebooks" else "./data",
                                       train=train, download=True, transform=t)
    return DataLoader(ds, batch_size=64, shuffle=train)

test_loader = get_loader_no_aug(train=False)
lr = 0.001
epochs = 15

# 无增强
model_no_aug = BaselineCNN().to(device)
opt = optim.Adam(model_no_aug.parameters(), lr=lr)
crit = nn.CrossEntropyLoss()
hist_no_aug = {"train": [], "test": []}
for ep in range(epochs):
    train_loss, train_acc = train_one_epoch(model_no_aug, get_loader_no_aug(train=True), opt, crit, device)
    test_acc = evaluate(model_no_aug, test_loader, device)
    hist_no_aug["train"].append(train_acc)
    hist_no_aug["test"].append(test_acc)
    print(f"NoAug Epoch {ep+1}: Train={train_acc:.1f}% Test={test_acc:.1f}%")

# 有增强
model_aug = BaselineCNN().to(device)
opt = optim.Adam(model_aug.parameters(), lr=lr)
hist_aug = {"train": [], "test": []}
for ep in range(epochs):
    train_loss, train_acc = train_one_epoch(model_aug, get_loaders_with_aug(), opt, crit, device)
    test_acc = evaluate(model_aug, test_loader, device)
    hist_aug["train"].append(train_acc)
    hist_aug["test"].append(test_acc)
    print(f"Aug Epoch {ep+1}: Train={train_acc:.1f}% Test={test_acc:.1f}%")

# %% [markdown]
# ## 3. 对比曲线

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist_no_aug["train"], label="No Aug Train")
axes[0].plot(hist_no_aug["test"], label="No Aug Test")
axes[0].set_title("Without Data Augmentation")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)"); axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot(hist_aug["train"], label="Aug Train")
axes[1].plot(hist_aug["test"], label="Aug Test")
axes[1].set_title("With Data Augmentation")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)"); axes[1].legend()
axes[1].grid(alpha=0.3)
fig.suptitle("Data Augmentation Effect on Generalization")
fig.tight_layout()

print(f"No Aug  - Final test acc: {hist_no_aug['test'][-1]:.1f}%, train-test gap: {hist_no_aug['train'][-1] - hist_no_aug['test'][-1]:.1f}%")
print(f"With Aug - Final test acc: {hist_aug['test'][-1]:.1f}%, train-test gap: {hist_aug['train'][-1] - hist_aug['test'][-1]:.1f}%")
