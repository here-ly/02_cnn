"""
01_env_inspect.py — 环境观测 notebook 的必须检查 cell。

这些 cell 用于验证环境正确性，每项都必须包含可直接运行的代码。
"""

# %% [markdown]
# ## 单步追踪：逐 action 检查棋盘变化

# %%
env3 = Game2048Wrapper()
env3.reset()
env3.render()
step = 0
while True:
    action = np.random.randint(4)
    print(f"--- Step {step+1}: action={['up','down','left','right'][action]} ---")
    state, reward, done = env3.step(action)
    env3.render()
    print(f"reward={reward:.0f}  done={done}")
    step += 1
    if done or step >= 200:
        break
print(f"Final: score={env3.get_score()}, max_tile={env3.get_max_tile()}")

# %% [markdown]
# ## 状态分布检查（数值范围 + NaN 检测）

# %%
states = []
for _ in range(1000):
    env.game.reset()
    states.append(env._get_state())
states = np.array(states)
print(f"state min={states.min():.2f} max={states.max():.2f} mean={states.mean():.2f}")
assert not np.isnan(states).any(), "NaN detected in states!"

# %% [markdown]
# ## Reward 分布检查

# %%
rewards = []
env.game.reset()
for _ in range(500):
    _, r, done = env.step(np.random.randint(4))
    rewards.append(r)
    if done:
        env.game.reset()
print(f"reward distribution: min={min(rewards)}, max={max(rewards)}, mean={np.mean(rewards):.2f}")

# %% [markdown]
# ## 各方向 valid 频率

# %%
valid_counts = [0] * 4
for _ in range(100):
    env.game.reset()
    for a in range(4):
        if env.get_valid_actions()[a]:
            valid_counts[a] += 1
for a, name in enumerate(["up", "down", "left", "right"]):
    print(f"  {name}: {valid_counts[a]} / 100")
