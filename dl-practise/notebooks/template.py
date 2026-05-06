# ---
# jupytext:
#   formats: ipynb,py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
# ---

# %% [markdown]
# # 01 探索性数据分析
#
# 本 notebook 用于：
# 1. 可视化数据分布
# 2. 检查类别平衡
# 3. 验证数据加载逻辑

# %%
import torch
import wandb
from pathlib import Path
from src.data.dataset import MyDataset

# %% [markdown]
# ## 1. 加载数据

# %%
dataset = MyDataset(data_dir="data/processed")
print(f"Dataset size: {len(dataset)}")

# %%
sample, label = dataset[0]
print(f"Sample shape: {sample.shape}, Label: {label}")
