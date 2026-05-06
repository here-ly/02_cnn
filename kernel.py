# Kaggle Git Clone entry point
import os, sys, subprocess, shutil, tarfile

REPO_URL = "https://github.com/here-ly/02_cnn.git"
BRANCH = "main"
WORKDIR = "/kaggle/working/repo"

print(f"Cloning {REPO_URL} ...")
subprocess.run(["git", "clone", "-b", BRANCH, "--depth", "1", REPO_URL, WORKDIR], check=True)
os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)

# ====== GPU 兼容 ======
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
    print("P100/K80 GPU detected — incompatible, falling back to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

# ====== CIFAR-10 数据（Kaggle dataset 或自动下载） ======
KAGGLE_DATA = "/kaggle/input/cifar10-extracted"
if os.path.isdir(KAGGLE_DATA):
    os.makedirs("data", exist_ok=True)
    for fname in os.listdir(KAGGLE_DATA):
        src = os.path.join(KAGGLE_DATA, fname)
        dst = os.path.join("data", "cifar-10-batches-py", fname) if not os.path.isdir(src) else None
        if os.path.isfile(src):
            os.makedirs("data/cifar-10-batches-py", exist_ok=True)
            shutil.copy2(src, os.path.join("data", "cifar-10-batches-py", fname))
    print("CIFAR-10 copied from Kaggle dataset.")
elif os.path.isdir("/kaggle/input/cifar10"):
    os.makedirs("data", exist_ok=True)
    for fname in os.listdir("/kaggle/input/cifar10"):
        src = os.path.join("/kaggle/input/cifar10", fname)
        dst = os.path.join("data", fname)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif fname.endswith(".tar.gz"):
            shutil.copy2(src, dst)
            with tarfile.open(dst, "r:gz") as tf:
                tf.extractall("data")
    print("CIFAR-10 extracted from Kaggle dataset.")

print("Installing dependencies ...")
subprocess.run([sys.executable, "-m", "pip", "install", "wandb", "--quiet"], check=False)

print(f"Running training in {WORKDIR} ...")
os.system(f"{sys.executable} scripts/train.py --config configs/kaggle.yaml")
