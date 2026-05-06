# Kaggle Git Clone entry point
import os, sys, subprocess, shutil

REPO_URL = "https://github.com/here-ly/02_cnn.git"
BRANCH = "main"
WORKDIR = "/kaggle/working/repo"
KAGGLE_DATA = "/kaggle/input/cifar10-data"

print(f"Cloning {REPO_URL} ...")
subprocess.run(["git", "clone", "-b", BRANCH, "--depth", "1", REPO_URL, WORKDIR], check=True)
os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)

# ====== GPU 兼容：import torch 前用 nvidia-smi 检测 ======
def _needs_cu118():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and ("P100" in r.stdout or "K80" in r.stdout or "M60" in r.stdout):
            return True
    except Exception:
        pass
    return False

if _needs_cu118():
    print("P100/K80 GPU detected — PyTorch 2.10 not compatible, falling back to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

# ====== CIFAR-10 数据 ======
if os.path.isdir(KAGGLE_DATA):
    os.makedirs("data", exist_ok=True)
    for fname in os.listdir(KAGGLE_DATA):
        src = os.path.join(KAGGLE_DATA, fname)
        dst = os.path.join("data", fname)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print("CIFAR-10 data copied from Kaggle dataset.")
else:
    print("WARNING: Kaggle dataset not found at", KAGGLE_DATA)

print("Installing dependencies ...")
subprocess.run([sys.executable, "-m", "pip", "install", "wandb", "--quiet"], check=False)

print(f"Running training in {WORKDIR} ...")
os.system(f"{sys.executable} scripts/train.py --config configs/kaggle.yaml")
