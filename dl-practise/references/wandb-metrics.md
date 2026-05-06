# Wandb 必监控指标

单项指标上涨不代表学得好，必须多指标交叉验证。

## 训练期指标（每 episode log 一次）

| 指标 | 含义 | hack 信号 |
|------|------|----------|
| `episode/return` | 累计奖励 | ↑ 但 max_tile 不涨 → 刷小分 |
| `episode/score` | 2048 总分 | 核心指标，但需配合其他 |
| `episode/max_tile` | 本局最大方块 | 真实进步的唯一硬指标 |
| `episode/length` | 存活步数 | ↓ + return ↑ → 刷无效分（不可能）；↓ + score ↓ → 策略变差 |
| `episode/epsilon` | 当前探索率 | 正常衰减即可 |
| `episode/invalid_rate` | 无效移动占比 | >0.5 → agent 在瞎按 |
| `episode/action_entropy` | 动作分布熵（0~1） | →0 → 卡死在单一方向，alarm |
| `action/U`, `action/D`, `action/L`, `action/R` | 各方向占比 | 肉眼直接看到偏斜 |
| `board/empty_cells` | game-over 时空位数 | →0 太快 = 策略差，空间耗尽快 |
| `board/avg_tile` | game-over 时平均 tile 值 | 越高越好，反映长期积累 |
| `board/merge_count` | 本局有效合并次数 | /episode_length = 有效操作率 |

## 评估期指标（每 N 个 episode eval 一次）

| 指标 | 含义 | hack 信号 |
|------|------|----------|
| `eval/avg_score` | 贪婪策略平均分 | 不涨反跌 → overfitting Q 值 |
| `eval/best_max_tile` | 最高 tile | 和训练期 max_tile 差距大 → 探索用得好但策略本身弱 |
| `eval/action_entropy` | 评估时动作熵 | 应为较低值（确定性策略）；→0 可能正常也可能死板 |
| `eval/avg_empty_cells` | 评估时空位余量 | 对比训练期 → 评估应更优 |

## Dashboard 推荐面板

```
Panel 1: episode/return  +  eval/avg_score     ← 双线: 训练vs评估
Panel 2: episode/max_tile  +  eval/best_max_tile
Panel 3: episode/action_entropy  +  eval/action_entropy
Panel 4: action/U D L R (stacked bar)           ← 一眼看偏斜
Panel 5: board/empty_cells  +  episode/invalid_rate
Panel 6: train/loss  +  episode/epsilon
```
