# Kaggle Git Clone entry point — template
import os, sys, subprocess, shutil, tarfile

REPO_URL = "https://github.com/用户名/仓库.git"
BRANCH = "main"
WORKDIR = "/kaggle/working/repo"

# ====== 1. Git Clone（已存在则 pull） ======
if os.path.isdir(WORKDIR):
    print("Repo exists, pulling latest ...")
    subprocess.run(["git", "-C", WORKDIR, "fetch", "--depth", "1", "origin", BRANCH], check=False)
    subprocess.run(["git", "-C", WORKDIR, "reset", "--hard", f"origin/{BRANCH}"], check=True)
else:
    print(f"Cloning {REPO_URL} ...")
    subprocess.run(["git", "clone", "-b", BRANCH, "--depth", "1", REPO_URL, WORKDIR], check=True)
os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)

# ====== 2. GPU 兼容（P100/K80/K40/M60 降级 CPU） ======
def _check_gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            old_gpus = ("P100", "K80", "K40", "M60")
            if any(g in r.stdout for g in old_gpus):
                return "fallback"
            return "gpu"
    except Exception:
        pass
    return "none"

gpu_status = _check_gpu()
if gpu_status == "fallback":
    print("Old GPU detected — falling back to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif gpu_status == "gpu":
    print("T4 or newer GPU detected — using CUDA.")
else:
    print("No GPU detected, using CPU.")

# ====== 3. 数据准备（优先 Kaggle Dataset，否则自动下载） ======
# 列出你引用的 Kaggle dataset 挂载路径
DATA_PATHS = ["/kaggle/input/your-dataset"]

for ds_path in DATA_PATHS:
    if os.path.isdir(ds_path):
        os.makedirs("data", exist_ok=True)
        for fname in os.listdir(ds_path):
            src = os.path.join(ds_path, fname)
            if os.path.isfile(src) and fname.endswith(".tar.gz"):
                shutil.copy2(src, os.path.join("data", fname))
                with tarfile.open(os.path.join("data", fname), "r:gz") as tf:
                    tf.extractall("data")
            elif os.path.isdir(src):
                shutil.copytree(src, os.path.join("data", fname), dirs_exist_ok=True)
            else:
                shutil.copy2(src, os.path.join("data", fname))
        print(f"Data loaded from {ds_path}")
        break
else:
    print("No Kaggle dataset found, will auto-download if needed.")

# ====== 4. Wandb（从 Kaggle Secrets 读取 API Key） ======
# 用户在 Kaggle → Add-ons → Secrets 添加 WANDB_API_KEY
if not os.environ.get("WANDB_API_KEY"):
    print("WARNING: WANDB_API_KEY not set. Add it in Kaggle → Add-ons → Secrets")

print("Installing dependencies ...")
subprocess.run([sys.executable, "-m", "pip", "install", "wandb", "--quiet"], check=False)

# ====== 5. 启动训练 ======
print(f"Running training ...")
os.system(f"{sys.executable} scripts/train.py --config configs/default.yaml")
