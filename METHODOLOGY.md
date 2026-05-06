# CIFAR-10 CNN 分类 —— 方法论

## 问题定义

对 CIFAR-10 数据集进行 10 类图像分类。输入为 32×32×3 的 RGB 图像，输出为 10 类的 logits。

| 要素  | 说明                                                                      |
| --- | ----------------------------------------------------------------------- |
| 输入  | 3×32×32 RGB 图像                                                          |
| 输出  | 10 类 logits（plane, car, bird, cat, deer, dog, frog, horse, ship, truck） |
| 训练集 | 50,000 张                                                                |
| 测试集 | 10,000 张                                                                |
| 类别  | 均衡分布，每类 5,000 训练 / 1,000 测试                                             |

## 公式

### Loss 函数

交叉熵损失（与 `nn.CrossEntropyLoss` 等价，内置 softmax）：

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{z_{i, y_i}}}{\sum_{j=1}^{10} e^{z_{i,j}}}
$$

### 数据预处理

归一化使用 CIFAR-10 的全局均值和标准差：

$$
x_{\text{norm}} = \frac{x - \mu}{\sigma}, \quad
\mu = (0.4914, 0.4822, 0.4465), \quad
\sigma = (0.2023, 0.1994, 0.2010)
$$

### 自适应学习率

基于测试 accuracy 曲线 AUC 增量调整学习率：

$$
\text{gain} = \max\left(0, \text{AUC}(y_{1:t}) - \text{AUC}(y_{1:t-1})\right)
$$

$$
\text{multiplier} = (1/4)^{\text{gain}}
$$

$$
\text{lr}_{\text{new}} = \beta \cdot \text{lr}_{\text{old}} + (1-\beta) \cdot \text{clip}(\text{lr}_{\text{old}} \cdot \text{multiplier}, \text{lr}_{\min}, \text{lr}_{\max})
$$

- 当 gain 很小（<0.05），multiplier 固定为 3.0，加速脱离平台期
- 当 gain 很大，lr 最多缩小到原来的 1/4

## 网络结构

```
Input (3×32×32)
  ├── Conv2d(3→32, k=3, p=1) + ReLU + MaxPool2d(2)  → 32×16×16
  ├── Conv2d(32→64, k=3, p=1) + ReLU + MaxPool2d(2) → 64×8×8
  ├── Conv2d(64→128, k=3, p=1) + ReLU + MaxPool2d(2) → 128×4×4
  ├── Flatten → 2048
  ├── Linear(2048 → 256) + ReLU
  └── Linear(256 → 10)
```

卷积通道数、kernel size、全连接隐藏维度均从 `configs/default.yaml` 读取，`_calc_conv_out()` 自动计算 FC 层输入维度。

## 设计决策

| 决策                                | 原因                                      |
| --------------------------------- | --------------------------------------- |
| 3 层卷积而非更深的网络                      | 配合作业要求，CIFAR-10 32×32 图像不需要太深           |
| RandomCrop + RandomHorizontalFlip | 经典 CIFAR-10 增强组合，有效减少过拟合                |
| 自适应 LR（基于 AUC 增益）                 | 避免手动调参，训练后期自动降温                         |
| 零硬编码                              | 所有超参在 `default.yaml`，网络结构也由配置驱动         |
| CPU 兼容                            | `get_device()` 自动检测 cuda/mps/cpu，无需修改代码 |

## 全链路说明

| 阶段          | 文件                         | 说明                                                      |
| ----------- | -------------------------- | ------------------------------------------------------- |
| 数据探索        | `notebooks/01_eda.py`      | 可视化数据分布、增强效果、归一化                                        |
| Baseline 对比 | `notebooks/02_baseline.py` | 有/无增强对比，网络结构对比                                          |
| 组件调试        | `notebooks/03_debug.py`    | 前向/反向/grad/ckpt 全链路验证                                   |
| 核心实现        | `src/models/backbone.py`   | 配置驱动的 CNN 模型                                            |
| 数据管道        | `src/data/transforms.py`   | 数据增强 + Dataset/DataLoader                               |
| 训练引擎        | `src/engine/trainer.py`    | 训练循环 + 自适应 LR + 可视化                                     |
| 指标收集        | `src/metrics/collector.py` | 混淆矩阵、梯度热力图、wandb 兼容                                     |
| 训练入口        | `scripts/train.py`         | `python scripts/train.py --config configs/default.yaml` |
| 评估入口        | `scripts/evaluate.py`      | `python scripts/evaluate.py --checkpoint xxx.pt`        |

### 运行命令

```bash
conda activate deep

# 快速验证（test_mode=true）
python scripts/train.py --config configs/default.yaml

# 正式训练（test_mode=false）
python scripts/train.py --config configs/default.yaml

# 评估
python scripts/evaluate.py --checkpoint outputs/checkpoints/final.pt
```
