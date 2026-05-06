# RL 预训检查（Pilot Validation）

RL 项目在正式训练前，必须先跑 `scripts/pilot.py`，确保环境和奖励设计没有明显漏洞。

## 四个检查

| 序号 | 检查 | 断言 | 不通过说明 |
|------|------|------|-----------|
| ① | 随机基线 | 20 局 → avg_score, best_tile | 环境可运行，baseline 值 |
| ② | 单向策略 vs 有 mask 随机 | 20 局单向(4方向) + 20 局 masked random | **masked random >> 单向** → masking 压制了无效动作 |
| ③ | 短训预检 | 500 步训练 | (a) loss 不爆炸 (b) 4 个方向占比 >1% (c) action_entropy >0.3 (d) epsilon 正常衰减 |
| ④ | 已训练 agent 评估 | 4 局贪婪评估 | 不 crash 即可（500 步不足以明显进步） |

## 使用

```bash
python scripts/pilot.py
# 全绿 [PASS] → 开始正式训练
# 有任何 [FAIL] → 检查 reward / env / network 设计
```

## 关键规则

- **action masking 是强制板机制**：policy 输出的 Q 值，在 select_action 时用 `action_mask` 把无效动作的 Q 设为 -inf，确保只有能改变棋盘的动作被选中。ε-greedy 探索也只在 valid actions 中随机选。
- **无效移动必须 reward < 0**：-1 惩罚防止 agent 学会 spam 单个方向。同时检测无效移动后在 step 中不 spawn tile 不改变棋盘。
- **Loss 下降 ≠ 策略好**：loss 只衡量 Q 值拟合程度，必须交叉看 eval 指标。
