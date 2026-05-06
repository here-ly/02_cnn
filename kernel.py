# Kaggle GPU Smoke Test — 合成数据（零网络依赖）
import os, sys, subprocess, time

REPO_URL = "https://github.com/here-ly/02_cnn.git"
WORKDIR = "/kaggle/working/repo"

print(f"Cloning {REPO_URL} ...")
subprocess.run(["git", "clone", "-b", "main", "--depth", "1", REPO_URL, WORKDIR], check=True)
os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)

# GPU 兼容检测（P100 降级 CPU）
def _needs_cu118():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and ("P100" in r.stdout or "K80" in r.stdout or "M60" in r.stdout):
            return True
    except Exception:
        pass
    return False

if _needs_cu118():
    print("P100 detected — falling back to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ====== 系统信息 ======
print(f"Torch: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} (SM {torch.cuda.get_device_capability(0)})")
print(f"Device: {device}")

# ====== 合成数据 ======
N, C, H, W, K = 2000, 3, 32, 32, 10
X = torch.randn(N, C, H, W)
y = torch.randint(0, K, (N,))
X_test = torch.randn(500, C, H, W)
y_test = torch.randint(0, K, (500,))
train_loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)
print(f"Data: {N} train / 500 test (synthetic)")

# ====== 模型 ======
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        return self.head(self.features(x))

model = TinyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# ====== 训练 2 epoch ======
t0 = time.time()
for epoch in range(2):
    model.train()
    running_loss = correct = total = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = model(inputs).max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    print(f"Epoch {epoch+1}/2: loss={running_loss/len(train_loader):.4f}  acc={100.*correct/total:.1f}%")

# 测试
model.eval()
correct = total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, pred = model(inputs).max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

print(f"Test acc: {100.*correct/total:.1f}%")
print(f"Total time: {time.time()-t0:.1f}s")
print("=== GPU Smoke Test PASSED! ===")
