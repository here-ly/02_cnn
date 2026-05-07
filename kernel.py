# Kaggle 训练入口 — Git Clone + wandb + 真实数据
import os, sys, subprocess, shutil, tarfile, warnings
warnings.filterwarnings("ignore")

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

# ====== CIFAR-10 数据（Kaggle Dataset 挂载，避免下载） ======
DS_DIR = "/kaggle/input/cifar10-extracted"
if os.path.isdir(DS_DIR):
    target = "data/cifar-10-batches-py"
    os.makedirs(target, exist_ok=True)
    for fname in os.listdir(DS_DIR):
        src = os.path.join(DS_DIR, fname)
        dst = os.path.join(target, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"CIFAR-10 files ready ({len(os.listdir(target))} files)")
else:
    print("Dataset not mounted, CIFAR-10 will download automatically (~30s)")

# ====== Wandb API Key（Script kernel 需用 kaggle_secrets 主动读取） ======
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret = user_secrets.get_secret("WANDB_API_KEY")
    if secret:
        os.environ["WANDB_API_KEY"] = secret
        print("WANDB_API_KEY loaded from Kaggle Secrets")
    else:
        print("WARNING: WANDB_API_KEY is empty in Kaggle Secrets")
except Exception:
    print("WARNING: Failed to load WANDB_API_KEY from Kaggle Secrets")

# ====== 依赖 ======
print("Installing dependencies ...")
subprocess.run([sys.executable, "-m", "pip", "install", "wandb", "--quiet"], check=False)

print(f"Running training in {WORKDIR} ...")
os.system(f"{sys.executable} scripts/train.py --config configs/kaggle.yaml")
