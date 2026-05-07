---
name: dl-practise
description: 个人深度学习/强化学习实践项目模板。提供标准化的DL/RL两套项目架构、Jupytext notebook编写规范（避免JSON语法错误）、以及torch + wandb + time tracking的代码约定。当用户开始新的深度学习或强化学习实践项目时使用此skill。
---

# DL/RL Practise

个人深度学习与强化学习实践项目的标准化脚手架。

## 触发条件

- 用户开始新的深度学习项目（CV、NLP、多模态等）
- 用户开始新的强化学习项目（DQN、PPO、SAC等）
- 用户需要搭建项目结构、写notebook、配置wandb
- 用户提到"DL实践""RL实践""深度学习项目""强化学习项目"

## 环境要求

始终使用 conda 的 `deep` 环境：

```bash
conda activate deep
```

如果环境不存在：立刻通知用户

---

## 一、DL 项目架构

```
my-ml-project/
├── data/
│   ├── raw/                    # 原始数据，永不修改
│   ├── processed/              # 清洗/预处理后的数据
│   └── external/               # 预训练权重等外部资源
├── notebooks/
│   ├── 01_eda.py               # Jupytext percent 格式
│   ├── 02_baseline.py
│   └── 03_debug.py
├── src/
│   ├── data/
│   │   ├── dataset.py          # torch.utils.data.Dataset 封装
│   │   ├── transforms.py       # 数据增强
│   │   └── preprocessing.py    # raw → processed 清洗逻辑
│   ├── models/
│   │   ├── backbone.py         # 骨干网络
│   │   └── heads.py            # 任务头（分类/回归等）
│   ├── losses/                 # loss 注册与管理
│   ├── metrics/
│   │   └── collector.py        # 指标收集 → 统一调用 wandb.log()
│   ├── utils/
│   │   ├── timer.py            # @timeit 装饰器 + Timer 上下文管理器
│   │   ├── seed.py             # random/numpy/torch/cuda 四种子统一
│   │   ├── device.py           # 自动检测 cuda/mps/cpu
│   │   └── logging.py          # 日志配置
│   └── engine/
│       ├── trainer.py          # 训练主循环
│       └── evaluator.py        # 评估逻辑（validation/test）
├── configs/
│   ├── default.yaml            # 所有参数（data/model/optim/train）
│   ├── experiment/             # 实验覆盖
│   └── wandb.yaml              # wandb 参数
├── scripts/
│   ├── train.py                # 入口：python scripts/train.py
│   ├── evaluate.py
│   └── sweep.py                # wandb sweep 启动
├── outputs/
│   ├── checkpoints/{run_id}/
│   ├── logs/{run_id}/
│   └── predictions/{run_id}/
├── tests/
├── .env.example  .gitignore  requirements.txt  README.md  METHODOLOGY.md
```

## 二、RL 项目架构

```
my-rl-project/
├── data/  notebooks/  outputs/  tests/  .env.example  .gitignore
├── src/
│   ├── envs/
│   │   ├── base.py             # 环境统一接口（reset/step/render/close）
│   │   ├── wrappers.py         # Gymnasium 兼容 wrapper
│   │   └── {custom}.py         # 自定义环境
│   ├── agents/
│   │   ├── base.py             # Agent 基类（act / save / load）
│   │   ├── policy.py           # Actor 策略网络
│   │   ├── value.py            # Critic 价值网络
│   │   └── dqn.py / ppo.py     # 具体算法实现
│   ├── buffers/
│   │   ├── replay.py           # 经验回放缓冲区（DQN/SAC）
│   │   └── rollout.py          # 在线 rollout 缓冲区（PPO）
│   ├── losses/
│   │   └── rl_losses.py        # policy_loss, value_loss, entropy_bonus
│   ├── metrics/
│   │   └── collector.py        # episode_return, success_rate, avg_steps
│   ├── utils/
│   │   ├── timer.py / seed.py / device.py / logging.py
│   │   └── normalizer.py       # RL常用：状态归一化（RunningMeanStd）
│   └── engine/
│       ├── rollout.py          # ① 数据采集：agent 与环境交互收集 trajectory
│       ├── trainer.py          # ② 策略改进：从 buffer 采样 → 计算 loss → 更新网络
│       └── evaluator.py        # ③ 表现评估：确定性运行 N 个 episode 统计指标
├── configs/
│   ├── default.yaml            # 唯一的配置文件，所有参数（env/agent/buffer/train）
│   ├── experiment/
│   └── wandb.yaml
├── scripts/
│   ├── train.py   evaluate.py   enjoy.py   pilot.py
└── outputs/
    ├── checkpoints/{run_id}/   logs/{run_id}/   videos/{run_id}/
```

### RL 三阶段循环

```
① rollout.py → ② trainer.py → ③ evaluator.py
    ↑               |               |
    └─── 采集足够数据后触发 update ──┘               │
                          │                          │
                          └──── 每 N 步触发评估 ──────────┘
```

---

## 核心约定

| 约定              | 说明                                                                       |
| --------------- | ------------------------------------------------------------------------ |
| `conda deep`    | 所有命令在 deep 环境下执行                                                         |
| `wandb`         | 所有实验必须接入 wandb 跟踪                                                        |
| `torch`         | 模型继承 `nn.Module`，数据集继承 `Dataset`                                         |
| `timer`         | 使用 `utils/timer.py` 跟踪各阶段耗时                                              |
| `seed`          | 使用 `utils/seed.py` 统一所有随机种子                                              |
| `device`        | 禁止硬编码 `.cuda()`，统一从 `utils/device.py` 获取                                 |
| `config`        | **所有数值必须在 `default.yaml` 中定义，代码零硬编码**。只加载 default.yaml + wandb.yaml 两份文件 |
| `single_config` | 构造函数统一接受 `config: dict`，内部 `cfg.get(key, default)` 取值                    |
| `pilot`         | **RL 项目：正式训练前必须先跑 `scripts/pilot.py`**，全绿再训练                             |
| `multi_metric`  | 单项 wandb 指标上涨不足以说明进步，必须多指标交叉验证                                           |

> 详细规约（构造方式、硬编码清单、checkpoint约定）见 [`references/conventions.md`](references/conventions.md)。

---

## 三、Notebook 编写规范

**禁止直接手写 `.ipynb` JSON 文件**。使用 Jupytext Percent 格式：

```python
# ---
# jupytext:
#   formats: ipynb,py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
# ---

# %% [markdown]
# # 01 探索性数据分析

# %%
import torch
from src.data.dataset import MyDataset

dataset = MyDataset(data_dir="data/processed")
print(f"Dataset size: {len(dataset)}")
```

规则：`# %% [markdown]` → Markdown cell，`# %%` → Code cell，头部必须有 YAML 头。VS Code 原生支持 `# %%` 逐 cell 运行。

**每个 notebook 必须包含可直接运行来排查 Bug 的检查 cell**（禁止只有 print 统计）。RL 三类 notebook 的必须检查见：

- [`notebooks/env-check-cells.py`](notebooks/env-check-cells.py)
- [`notebooks/baseline-check-cells.py`](notebooks/baseline-check-cells.py)
- [`notebooks/training-check-cells.py`](notebooks/training-check-cells.py)

```bash
jupytext --to notebook notebooks/01_eda.py          # 生成 ipynb
jupytext --set-formats ipynb,py:percent *.py        # 批量双向配对
```

---

## 四、工作流程

```
notebooks/*.py          src/                    scripts/train.py
(探索 & 调试)    →    (核心逻辑实现)    →       (正式训练)
```

1. **Notebook**：探索数据、调试模型、验证想法
2. **迁移**：确认无误后迁入 `src/`
3. **配置**：在 `configs/default.yaml` 中定义超参
4. **本地训练**：`python scripts/train.py --config configs/experiment/exp001.yaml`
5. **远程训练**：推送 Kaggle（见下方）
6. **查看**：wandb dashboard 对比实验
7. **必含**：项目根目录必须有 `METHODOLOGY.md`（模板见 [`references/methodology-template.md`](references/methodology-template.md)）

### Kaggle 远程训练

GPU 锁定必须在 GUI 操作，**工作流：CLI 改参数 + push → GUI 选 GPU 跑**。

**硬件**：

| 资源 | 规格 |
|------|------|
| GPU T4×2 | 2× Tesla T4, 16GB/卡, SM 7.5, 总计 32GB VRAM |
| GPU P100 | SM 6.0, PyTorch 2.10 不兼容 → kernel.py 自动降级 CPU |
| CPU | Intel Xeon @ 2.00GHz, 4核8线程, 30GB RAM |

**一键推送**（改参数+git push+GUI指南）：

```powershell
.\scripts\kaggle.ps1 -Lr 0.0003 -Epochs 100 -WandbMode offline
.\scripts\kaggle.ps1 -Show    # 查看当前参数
.\scripts\kaggle.ps1 -NoPush  # 只改参数不push
```

**GUI 挂载清单**（Settings → Internet ON → Accelerator → GPU T4 x2，然后 Add Data）：

| 数据集 | 挂载后路径 | 用途 |
|--------|-----------|------|
| `herely/{project}-data` | `/kaggle/input/{slug}/` | 训练数据（秒加载，避免下载） |
| `herely/wandb-key` (私有) | `/kaggle/input/wandb-key/` | WANDB_API_KEY |

**参数控制**：所有配置在 `configs/kaggle_full.yaml`，代码零硬编码：
```yaml
wandb:
  mode: "offline"    # 每epoch快3-5s，训练完 wandb sync 上传
data:
  num_workers: 2     # CPU 用 2；T4 GPU 改 8
```

详细指南见 [`references/kaggle-workflow.md`](references/kaggle-workflow.md)。

**快速命令**：
```bash
git push && kaggle kernels push -p .              # 纯CLI推送
kaggle kernels status 用户名/项目名                 # 状态
kaggle kernels logs 用户名/项目名                    # 实时日志
kaggle kernels output 用户名/项目名 -p ./outputs/    # 下载产物

详细指南见 [`references/kaggle-workflow.md`](references/kaggle-workflow.md)。

---

## 快速命令

```bash
conda activate deep
python scripts/pilot.py                    # RL预检（必须）
python scripts/train.py --config configs/experiment/exp001.yaml
python scripts/evaluate.py --checkpoint outputs/checkpoints/xxx/epoch_50.pt
wandb sweep configs/sweep.yaml
python -m pytest tests/ -v
```

---

## 文件索引

| 路径                                                                         | 内容                                |
| -------------------------------------------------------------------------- | --------------------------------- |
| [`templates/dl/train.py`](templates/dl/train.py)                           | DL 训练脚本模板                         |
| [`templates/rl/train.py`](templates/rl/train.py)                           | RL 训练脚本模板                         |
| [`templates/rl/agent_dqn.py`](templates/rl/agent_dqn.py)                   | DQNAgent 完整实现                     |
| [`templates/rl/network_cnn.py`](templates/rl/network_cnn.py)               | CNNQNetwork 实现                    |
| [`templates/dl/default.yaml`](templates/dl/default.yaml)                   | DL 配置示例                           |
| [`templates/rl/default.yaml`](templates/rl/default.yaml)                   | RL 配置示例                           |
| [`templates/shared/`](templates/shared/)                                   | 共享工具（timer/seed/device/collector/kernel） |
| [`scripts/update_config.py`](scripts/update_config.py)                     | YAML 参数 CLI 更新工具                       |
| [`scripts/kaggle.ps1`](scripts/kaggle.ps1)                                 | Kaggle 一键推送 + GUI 指南 PowerShell 脚本      |
| [`notebooks/`](notebooks/)                                                 | Notebook 模板及检查 cell 参考            |
| [`references/conventions.md`](references/conventions.md)                   | 配置驱动设计详细规约                        |
| [`references/wandb-metrics.md`](references/wandb-metrics.md)               | Wandb 监控指标全表                      |
| [`references/rl-pilot.md`](references/rl-pilot.md)                         | RL Pilot 预检流程                     |
| [`references/methodology-template.md`](references/methodology-template.md) | METHODOLOGY.md 模板                 |
| [`references/kaggle-workflow.md`](references/kaggle-workflow.md)           | Kaggle 远程训练完整指南（GPU/数据/wandb/排坑） |
