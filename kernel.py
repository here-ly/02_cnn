# Kaggle 训练入口 — Git Clone + wandb + 真实数据
import os, sys, subprocess, shutil, tarfile

REPO_URL = "https://github.com/here-ly/02_cnn.git"
BRANCH = "main"
WORKDIR = "/kaggle/working/repo"

if os.path.isdir(WORKDIR):
    print("Repo exists, pulling latest ...")
    subprocess.run(["git", "-C", WORKDIR, "fetch", "--depth", "1", "origin", BRANCH], check=False)
    subprocess.run(["git", "-C", WORKDIR, "reset", "--hard", f"origin/{BRANCH}"], check=True)
else:
    print(f"Cloning {REPO_URL} ...")
    subprocess.run(["git", "clone", "-b", BRANCH, "--depth", "1", REPO_URL, WORKDIR], check=True)
os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)

# ====== GPU 兼容（P100/K80 降级 CPU） ======
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
    print("P100/K80 detected — falling back to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ====== CIFAR-10 数据（从 Kaggle Dataset 或自动下载） ======
for ds_path in ["/kaggle/input/cifar10-extracted", "/kaggle/input/cifar10", "/kaggle/input/cifar10-data"]:
    if os.path.isdir(ds_path):
        os.makedirs("data", exist_ok=True)
        for fname in os.listdir(ds_path):
            src = os.path.join(ds_path, fname)
            if os.path.isfile(src) and fname.endswith(".tar.gz"):
                shutil.copy2(src, os.path.join("data", fname))
                with tarfile.open(os.path.join("data", fname), "r:gz") as tf:
                    tf.extractall("data")
                print(f"CIFAR-10 extracted from {ds_path}")
            elif os.path.isdir(src):
                shutil.copytree(src, os.path.join("data", fname), dirs_exist_ok=True)
            else:
                shutil.copy2(src, os.path.join("data", fname))
        print(f"CIFAR-10 data ready from {ds_path}")
        break
else:
    print("No Kaggle dataset found, CIFAR-10 will download automatically (~30s)")

# ====== Wandb API Key（Kaggle Secrets 自动注入为环境变量） ======
if not os.environ.get("WANDB_API_KEY"):
    print("WARNING: WANDB_API_KEY not set. Add it in Kaggle → Add-ons → Secrets")

# ====== 依赖 ======
print("Installing dependencies ...")
subprocess.run([sys.executable, "-m", "pip", "install", "wandb", "--quiet"], check=False)

print(f"Running training in {WORKDIR} ...")
os.system(f"{sys.executable} scripts/train.py --config configs/kaggle.yaml")
