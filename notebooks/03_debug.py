# ---
# jupytext:
#   formats: ipynb,py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
# ---

# %% [markdown]
# # 03 训练调试与组件验证
#
# 本 notebook 用于逐组件调试训练 pipeline：
# 1. 网络前向传播形状验证
# 2. 单 batch 训练（loss + backward 正常）
# 3. 梯度流检查
# 4. 学习率调度验证
# 5. checkpoint 保存/加载

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from src.utils.device import get_device
from src.data.transforms import build_datasets, build_loaders
from src.models.backbone import build_model
from src.losses.registry import build_loss
from src.engine.trainer import Trainer

# %%
device = get_device()
print(f"Device: {device}")

with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)

train_dataset, test_dataset = build_datasets(data_root="./data", use_augmentation=True)
train_loader, test_loader = build_loaders(
    train_dataset, test_dataset, batch_size=config["data"]["batch_size"], test_mode=False,
)

# %% [markdown]
# ## 1. 网络前向传播形状验证

# %%
model = build_model(config).to(device)
images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)

model.train()
outputs = model(images)
print(f"Input shape: {images.shape}")
print(f"Output shape: {outputs.shape}")
print(f"Expected output shape: [{config['data']['batch_size']}, {config['model']['num_classes']}]")
assert outputs.shape == (config["data"]["batch_size"], config["model"]["num_classes"]), "Output shape mismatch!"
print("PASS: Forward pass output shape correct")

# %% [markdown]
# ## 2. 单 batch 训练（验证 loss + backward）

# %%
criterion = build_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["optim"]["lr"])

loss = criterion(outputs, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss value: {loss.item():.4f}")
assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf!"
print("PASS: Loss is valid, backward + optimizer.step() succeeded")

# %% [markdown]
# ## 3. 梯度流检查

# %%
grad_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norms[name] = param.grad.norm().item()

print("Gradient norms:")
for name, norm in sorted(grad_norms.items()):
    print(f"  {name:40s}: {norm:.6f}")

has_zero_grad = any(norm == 0 for norm in grad_norms.values())
if has_zero_grad:
    print("WARNING: Some parameters have zero gradients")
else:
    print("PASS: All parameters have non-zero gradients")

# %% [markdown]
# ## 4. 自适应学习率调度验证

# %%
from src.engine.trainer import update_learning_rate, compute_auc_gain

test_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lr_history = [0.01]
acc_history = []

for i in range(20):
    acc = 30 + i * 2.5 + (i % 3) * 0.5
    acc_history.append(acc)
    if len(acc_history) >= 2:
        gain = compute_auc_gain(acc_history)
    else:
        gain = 0.0
    lr = update_learning_rate(test_optimizer, gain, lr_history[-1])
    lr_history.append(lr)

print("LR schedule (first 10 values):", [f"{x:.6f}" for x in lr_history[:10]])
assert all(not (lr == 0 or lr != lr) for lr in lr_history), "Invalid LR value!"
print("PASS: Learning rate schedule works")

# %% [markdown]
# ## 5. Checkpoint 保存/加载

# %%
import tempfile, os

tmpdir = tempfile.mkdtemp()
ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")

model.eval()
torch.save(model.state_dict(), ckpt_path)

model2 = build_model(config).to(device)
model2.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
model2.eval()

with torch.no_grad():
    out1 = model(images)
    out2 = model2(images)
    diff = (out1 - out2).abs().max().item()

print(f"Max output difference after load: {diff:.10f}")
assert diff < 1e-5, "Checkpoint load mismatch!"
print("PASS: Checkpoint save/load works correctly")
