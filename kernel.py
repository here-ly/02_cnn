# Kaggle Git Clone entry point
# kaggle kernels push 后自动执行此脚本

import os
import sys
import subprocess

REPO_URL = "https://github.com/here-ly/02_cnn.git"
BRANCH = "main"
WORKDIR = "/kaggle/working/repo"

print(f"Cloning {REPO_URL} ...")
subprocess.run(["git", "clone", "-b", BRANCH, "--depth", "1", REPO_URL, WORKDIR], check=True)
os.chdir(WORKDIR)

sys.path.insert(0, WORKDIR)

print("Installing dependencies ...")
subprocess.run([sys.executable, "-m", "pip", "install", "wandb", "--quiet"], check=False)

print(f"Running training in {WORKDIR} ...")
os.system(f"{sys.executable} scripts/train.py --config configs/default.yaml")
