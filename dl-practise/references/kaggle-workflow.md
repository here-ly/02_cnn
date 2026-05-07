# Kaggle 远程训练完整指南

## 硬件规格

| 资源 | 规格 | 说明 |
|------|------|------|
| GPU T4×2 | 2× Tesla T4, 16GB/卡, SM 7.5 | PyTorch 2.10+ cu128 原生支持 |
| GPU P100 | Tesla P100, 12GB, SM 6.0 | PyTorch 2.10 不支持，需降级 CPU |
| CPU | Intel Xeon @ 2.00GHz, 4核8线程 | 服务器级，CIFAR-10 可纯 CPU 跑 |
| RAM | 30GB | - |
| 免费配额 | 30h/week GPU | 每周一 UTC 重置 |

## T4 锁卡

Kaggle GPU 是**随机分配**的。Settings → Accelerator → **GPU T4 x2** → 点一次 Run，之后 `kaggle kernels push` 会保留该选择。但无法 100% 保证，高峰期可能降级 P100。

**kernel.py 内置保护**：
```python
# import torch 之前检测 GPU
r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], ...)
if "P100" in r.stdout or "K80" in r.stdout:  # SM < 7.0
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 降级 CPU
```

## 数据集策略

**方案对比**：

| 方案 | 速度 | 可靠性 | 适用 |
|------|------|--------|------|
| Kaggle Dataset 挂载 | 秒级（文件复制） | ★★★★★ | 推荐长期使用 |
| torchvision 自动下载 | ~30s (CIFAR-10) | ★★★★ | 首次快速验证 |
| 私有 Dataset 传 key | 秒级（读文本） | ★★★★★ | WANDB_API_KEY 等 |

**创建自己的数据集**：
```bash
# 1. 准备数据目录（含 dataset-metadata.json）
mkdir cifar10-dataset
cp cifar-10-batches-py/* cifar10-dataset/

# 2. 上传
kaggle datasets create -p cifar10-dataset --dir-mode zip

# 3. 在 kernel-metadata.json 中引用
"dataset_sources": ["herely/your-dataset"]
```

## WANDB_API_KEY

**Script kernel 不支持 `kaggle_secrets` 模块**（`ConnectionError`）。唯一可靠方案：**私有 Kaggle Dataset**。

```
# 创建私有 dataset 存放 key
echo "your_wandb_key" > wandb_api_key.txt
kaggle datasets create -p . --dir-mode zip
# → herely/wandb-key (private)

# kernel-metadata.json
"dataset_sources": ["herely/wandb-key"]

# kernel.py 读取
with open("/kaggle/input/wandb-key/wandb_api_key.txt") as f:
    os.environ["WANDB_API_KEY"] = f.read().strip()
```

## Wandb 加速

**Online 模式每 epoch 上传日志 (~3-5s)**，100 epoch 浪费 5-8 分钟。

**Offline 模式**：本地记录，训练完一次性上传：
```yaml
# configs/wandb.yaml
mode: "offline"
```
```bash
# 训练结束后
wandb sync ./wandb/offline-run-*
```

**Kernel.py 中设置**（在 pip install wandb 之后）：
```python
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])
```

## GUI vs CLI

| 操作 | CLI (`kaggle kernels push`) | GUI (网页 Run) |
|------|---------------------------|----------------|
| 代码更新 | `git push && kaggle kernels push` | 网页编辑器 |
| 数据集挂载 | `dataset_sources` 自动 | 右侧 Add Data 手动添加 |
| GPU 选择 | 继承上次设置 | Settings → Accelerator |
| 日志查看 | `kaggle kernels logs` | 网页 Logs tab |
| Wandb key | 挂载私有 dataset | 同左，或 Add-ons → Secrets |

**GUI 运行前检查清单**：
- [ ] Settings → Internet ON
- [ ] Settings → Accelerator → GPU T4 x2
- [ ] Add Data → `herely/{project}-data`
- [ ] Add Data → `herely/wandb-key` (私有)

## 快速 PS1 命令

```powershell
# 查看状态
kaggle kernels status herely/cifar10-cnn-train2

# 实时日志（script kernel 有效）
kaggle kernels logs herely/cifar10-cnn-train2

# 修改配置 + 推送
git add configs/kaggle_full.yaml; git commit -m "tune params"; git push
kaggle kernels push -p .

# 下载 checkpoint
kaggle kernels output herely/cifar10-cnn-train2 -p ./outputs/

# 同步 wandb 离线日志
wandb sync ./wandb/offline-run-*
```

## 常见坑

| 问题 | 原因 | 解决 |
|------|------|------|
| P100 `no kernel image` | SM 6.0 不兼容 PyTorch 2.10 | kernel.py nvidia-smi 检测 + CPU 降级 |
| Wandb key 读不到 | Script kernel 无 `kaggle_secrets` | 私有 Kaggle Dataset 挂载 |
| 数据集不挂载 | `dataset_sources` 只在 CLI push 生效 | GUI 需手动 Add Data |
| GPU 利用率 20% | CPU 数据加载瓶颈 | `num_workers: 6-8` |
| Epoch 10s 但计算 1.5s | Wandb online 同步开销 | 切 `mode: offline` |
| `git clone` 失败 | Re-run 时目录残留 | `git pull` fallback |
| `.gitignore` 误伤 src | `data/` 匹配 `src/data/` | 改为 `/data/` |
