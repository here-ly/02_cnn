# Kaggle Git Clone entry point
# kaggle kernels push 后自动执行此脚本

import os, sys, subprocess, shutil

REPO_URL = "https://github.com/here-ly/02_cnn.git"
BRANCH = "main"
WORKDIR = "/kaggle/working/repo"
KAGGLE_DATA = "/kaggle/input/cifar10-data"

print(f"Cloning {REPO_URL} ...")
subprocess.run(["git", "clone", "-b", BRANCH, "--depth", "1", REPO_URL, WORKDIR], check=True)
os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)

# ====== GPU 兼容检查（P100 需要 cu118 版 torch） ======
import torch as _t
if _t.cuda.is_available():
    sm = _t.cuda.get_device_capability(0)
    if sm < (7, 0):
        print(f"GPU sm_{sm[0]}.{sm[1]} requires cu118 PyTorch, installing ...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu118",
            "--quiet",
        ])

import torch  # noqa: E402

# ====== CIFAR-10 数据：从 Kaggle Dataset 复制到 data/ ======
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
