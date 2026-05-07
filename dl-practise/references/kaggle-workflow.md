# Kaggle 远程训练完整指南

## 前置准备（一次）

```bash
pip install kaggle
# 从 https://www.kaggle.com/settings/account → Create New Token
# 将 kaggle.json 放到 ~/.kaggle/

# GitHub CLI（推荐，可选）
winget install GitHub.cli  # 或 chocolatey: choco install gh
gh auth login
```

## 项目初始化

每个想推送 Kaggle 的项目需要两个文件：

- `kernel.py` — 训练入口脚本，负责 clone/pull、GPU 检测、数据准备、wandb 注入、启动训练
- `kernel-metadata.json` — Kaggle kernel 元数据

模板见 [`templates/shared/kernel.py`](../templates/shared/kernel.py) 和 [`templates/shared/kernel-metadata.json`](../templates/shared/kernel-metadata.json)。

## 日常工作流

```bash
# 本地改代码
git add -A && git commit -m "xxx" && git push

# 推送到 Kaggle
kaggle kernels push -p .

# 查看状态
kaggle kernels status 用户名/项目名

# 查看实时日志（仅 script 类型 kernel 有效）
kaggle kernels logs 用户名/项目名

# 下载产物（checkpoint、图片等）
kaggle kernels output 用户名/项目名 -p ./outputs/
```

## GPU 选型

| GPU | SM | Kaggle 免费池 | PyTorch 2.10 兼容 |
|-----|-----|------------|-----------------|
| T4 x2 | 7.5 | 需手动锁 | ✅ 原生支持 |
| P100 | 6.0 | 默认分配 | ❌ 需 CPU 降级 |

**锁定 T4**：在 Kaggle web 页面 → Settings → Accelerator → GPU T4 x2 → 手动 Run 一次。之后 `kaggle kernels push` 会保留该设置。

**P100 降级**：`kernel.py` 模板已内置 nvidia-smi 检测，P100 自动设 `CUDA_VISIBLE_DEVICES=""` 降级 CPU，不会 crash。

## Script vs Notebook

| | Script | Notebook |
|------|--------|----------|
| CLI 日志 (`kaggle kernels logs`) | ✅ 实时 | ❌ 不可靠 |
| 网页交互编辑 | ❌ | ✅ |
| 进程稳定性 | ✅ 一个进程到底 | ❌ 可能重启 |
| **推荐用途** | **正式训练** | 交互实验 |

## Wandb 集成

1. Kaggle Secrets：在 kernel 页面 → Add-ons → Secrets → 添加 `WANDB_API_KEY`
2. `kernel.py` 中 `os.environ.get("WANDB_API_KEY")` 会自动读取
3. `scripts/train.py` 中已有 `wandb.init()`，无需额外代码

## 数据策略

按优先级：
1. **Kaggle Dataset 挂载**（最快，0s）：上传数据为 Kaggle Dataset，在 `dataset_sources` 中引用，`kernel.py` 从 `/kaggle/input/` 复制到 `./data/`
2. **torchvision 自动下载**（次选，~30s）：CIFAR-10/MNIST 等标准数据集，`download=True` 自动从源站下载
3. **上传提取好的文件**（推荐长期）：把 `.bin` 文件直接上传为 dataset，避免 tar.gz 解压

## 常见坑

| 问题 | 原因 | 解决 |
|------|------|------|
| `destination path already exists` | 网页 Re-run 时目录未清理 | `kernel.py` 用 `git pull` 代替 `git clone` |
| P100 `no kernel image` 错误 | PyTorch 2.10 不支持 SM 6.0 | `kernel.py` 内置 nvidia-smi 检测 + CPU 降级 |
| CIFAR-10 下载 5min+ | Kaggle 到 cs.toronto.edu 慢 | 上传为 Kaggle Dataset 或接受 30s |
| Wandb 不记录 | API key 未设置 | Kaggle Secrets 中加 `WANDB_API_KEY` |
| `git clone` 分支不存在 | 本地 `master` vs 远程 `main` | 统一用 `main`；`kernel.py` 指定分支名 |
| `.gitignore` 吃掉 `src/data/` | 根级 `data/` 匹配了 `src/data/` | 改为 `/data/` |
