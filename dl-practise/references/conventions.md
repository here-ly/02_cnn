# 配置驱动设计规约

## 铁律

**代码中不允许出现任何硬编码数值。** 所有数值——包括网络结构的层数、通道数、kernel size——必须在 YAML 中定义。

## 构造方式

```python
# ✅ 正确：一个 config dict 传入，内部用 cfg.get() 读取
agent = DQNAgent(
    n_actions=env.n_actions,
    config=config["agent"],     # 网络结构 + lr/gamma/epsilon 全在这里
    device=device,
)

# ❌ 错误：逐参数传入（网络结构、lr 散落在外层）
agent = DQNAgent(lr=0.001, gamma=0.99, epsilon_start=1.0, ..., device=device)
```

## Agent 构造函数签名

所有 Agent / Trainer / Evaluator 构造函数统一接受 `config: dict` 参数，内部用 `cfg.get(key, default)` 读取：

```python
class DQNAgent:
    def __init__(self, n_actions: int, config: dict = None, device=None):
        cfg = config or {}
        self.gamma = cfg.get("gamma", 0.99)
        self.lr = cfg.get("lr", 0.001)
        self.epsilon = cfg.get("epsilon_start", 1.0)
        ...
        model_cfg = cfg.get("model", {})
        self.q_network = CNNQNetwork(
            n_actions=n_actions,
            conv_channels=model_cfg.get("conv_channels"),
            conv_kernels=model_cfg.get("conv_kernels"),
            fc_hidden=model_cfg.get("fc_hidden", 256),
        ).to(self.device)
```

## Checkpoint 含 config

保存时把 config 序列化进 ckpt，评估时无需另外指定 config 文件：

```python
agent.save(path, agent_config=config["agent"])
# evaluate:
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
agent = DQNAgent(config=ckpt.get("agent_config", {}), device=device)
agent.load(args.checkpoint)
```

> PyTorch 2.6+ 注意：`torch.load` 默认 `weights_only=True`，但 ckpt 中存了 numpy scalar 时会报错。加载自己保存的 ckpt（来源可信），传 `weights_only=False`。

## 配置结构要求

- 所有数值在 **一份** `default.yaml` 中定义，禁止拆分成 agent.yaml / env.yaml 等碎片文件
- 网络结构（type, hidden_dims / conv_channels, conv_kernels, fc_hidden）写在 `agent.model` 节
- 训练只加载 `default.yaml` + `wandb.yaml` 两份文件

## 硬编码检查清单

- [ ] 代码中没有 `nn.Conv2d(1, 128, 2)` 这种写死数字
- [ ] 代码中没有 `Linear(512, 256)` 这种写死维度（用 `_calc_conv_out()` 自动算）
- [ ] 代码中没有 `for _ in range(4)` 这种写死 update 次数
- [ ] 代码中没有 `buffer = ReplayBuffer(100000)` 这种写死容量
- [ ] 代码中没有 `lr=0.001, gamma=0.99` 这种直接传参
- [ ] `agent.save()` 调用时传入了 `agent_config=config["agent"]`
