# Kaggle Git Clone entry point
import os, sys, subprocess, shutil, warnings
warnings.filterwarnings("ignore")

# ====== CONFIGURATION (edit these) ======
REPO_URL = "https://github.com/用户名/仓库.git"
BRANCH = "main"
WORKDIR = "/kaggle/working/repo"
PROJECT_DATA_SLUG = "用户名/project-data"       # Kaggle dataset slug
WANDB_KEY_SLUG = "用户名/wandb-key"             # private dataset with API key
# ========================================

# 1. Git Clone（已存在则 pull）
if os.path.isdir(WORKDIR):
    print("Repo exists, pulling latest ...")
    subprocess.run(["git", "-C", WORKDIR, "fetch", "--depth", "1", "origin", BRANCH], check=False)
    subprocess.run(["git", "-C", WORKDIR, "reset", "--hard", f"origin/{BRANCH}"], check=True)
else:
    print(f"Cloning {REPO_URL} ...")
    subprocess.run(["git", "clone", "-b", BRANCH, "--depth", "1", REPO_URL, WORKDIR], check=True)
os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)

# 2. GPU 兼容（P100/K80/K40/M60 → CPU）
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

# 3. WANDB_API_KEY（从私有 Kaggle Dataset 读取）
key = os.environ.get("WANDB_API_KEY")
if not key:
    key_file = "/kaggle/input/wandb-key/wandb_api_key.txt"
    if os.path.isfile(key_file):
        with open(key_file, "r") as f:
            key = f.read().strip()
        if key:
            os.environ["WANDB_API_KEY"] = key
            print("WANDB_API_KEY loaded from private dataset")
    if not key:
        try:
            from kaggle_secrets import UserSecretsClient
            _val = UserSecretsClient().get_secret("WANDB_API_KEY")
            if _val:
                key = str(_val).strip()
                os.environ["WANDB_API_KEY"] = key
                print("WANDB_API_KEY loaded from Kaggle Secrets")
        except Exception:
            pass
if not key:
    print("WARNING: No WANDB_API_KEY found — wandb logging disabled")

# 4. 项目数据（从 Kaggle Dataset 挂载复制到 ./data/）
DS_DIR = "/kaggle/input/" + PROJECT_DATA_SLUG.split("/")[-1]
if os.path.isdir(DS_DIR):
    os.makedirs("data", exist_ok=True)
    for fname in os.listdir(DS_DIR):
        src = os.path.join(DS_DIR, fname)
        dst = os.path.join("data", fname)
        if os.path.isfile(src):
            if fname.endswith(".tar.gz"):
                import tarfile
                shutil.copy2(src, dst)
                with tarfile.open(dst, "r:gz") as tf:
                    tf.extractall("data")
            elif not os.path.exists(dst):
                shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Data loaded from {DS_DIR}")
else:
    print(f"Dataset {PROJECT_DATA_SLUG} not mounted, will auto-download if needed.")

# 5. 依赖安装
print("Installing dependencies ...")
subprocess.run([sys.executable, "-m", "pip", "install", "wandb", "--quiet"], check=False)
if os.environ.get("WANDB_API_KEY"):
    try:
        import wandb
        wandb.login(key=os.environ["WANDB_API_KEY"])
    except Exception:
        pass

# 6. 启动训练
print(f"Running training ...")
os.system(f"{sys.executable} scripts/train.py --config configs/kaggle_full.yaml")
