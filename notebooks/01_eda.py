# ---
# jupytext:
#   formats: ipynb,py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
# ---

# %% [markdown]
# # 01 CIFAR-10 探索性数据分析
#
# 本 notebook 用于：
# 1. 加载 CIFAR-10 数据集，检查数据形状和类型
# 2. 可视化样本图片，查看各类别分布
# 3. 验证数据增强效果
# 4. 检查像素值范围和归一化结果

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# %%
BATCH_SIZE = 64
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="../data" if Path.cwd().name == "notebooks" else "./data",
    train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="../data" if Path.cwd().name == "notebooks" else "./data",
    train=False, download=True, transform=transform
)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")
print(f"Classes: {classes}")

# %% [markdown]
# ## 1. 类别分布检查

# %%
train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
class_counts = np.bincount(train_labels)
for i, name in enumerate(classes):
    print(f"  {name}: {class_counts[i]}")
print(f"\nAll equal: {all(c == 5000 for c in class_counts)}")

# %% [markdown]
# ## 2. 可视化样本图片

# %%
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for cls in range(10):
    idx = train_labels.index(cls)
    img, _ = train_dataset[idx]
    ax = axes[cls // 5, cls % 5]
    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(classes[cls])
    ax.axis("off")
fig.tight_layout()

# %% [markdown]
# ## 3. 像素值分布检查

# %%
all_pixels = []
for i in range(1000):
    img, _ = train_dataset[i]
    all_pixels.append(img.numpy().flatten())
all_pixels = np.concatenate(all_pixels)
print(f"Pixel range: [{all_pixels.min():.4f}, {all_pixels.max():.4f}]")
print(f"Pixel mean: {all_pixels.mean():.4f}, std: {all_pixels.std():.4f}")

# %% [markdown]
# ## 4. 数据增强效果可视化

# %%
aug_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

sample_idx = 0
img, label = train_dataset[sample_idx]
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    from PIL import Image as PILImage
    pil_img = transforms.ToPILImage()(train_dataset.data[sample_idx])
    aug_img = aug_transform(pil_img)
    ax = axes[i // 4, i % 4]
    ax.imshow(aug_img.permute(1, 2, 0))
    ax.set_title(f"Aug #{i + 1}")
    ax.axis("off")
fig.suptitle(f"Data Augmentation for class: {classes[label]}")
fig.tight_layout()

# %% [markdown]
# ## 5. 归一化前后对比

# %%
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
normalize = transforms.Normalize(mean, std)

img_raw, _ = train_dataset[0]
img_norm = normalize(img_raw.clone())

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(img_raw.permute(1, 2, 0))
axes[0].set_title("Before normalization")
axes[0].axis("off")
axes[1].imshow(img_norm.permute(1, 2, 0))
axes[1].set_title("After normalization")
axes[1].axis("off")
fig.tight_layout()

print(f"Before norm: min={img_raw.min():.2f}, max={img_raw.max():.2f}")
print(f"After  norm: min={img_norm.min():.2f}, max={img_norm.max():.2f}")
