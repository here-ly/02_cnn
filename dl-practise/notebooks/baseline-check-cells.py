"""
02_random_baseline.py — 随机基线 notebook 的必须检查 cell。

用于建立"什么都不学"的基线，对比衡量智能体的真实提升。
"""

# %% [markdown]
# ## 随机基线（有 mask）：只在有效动作中随机

# %%
env = Game2048Wrapper()
scores, tiles, lengths = [], [], []
for _ in range(200):
    env.reset()
    ep_len = 0
    while True:
        valid = np.array(env.get_valid_actions(), dtype=bool)
        a = np.random.choice(np.where(valid)[0])
        _, _, done = env.step(a)
        ep_len += 1
        if done or ep_len >= 5000:
            break
    scores.append(env.get_score())
    tiles.append(env.get_max_tile())
    lengths.append(ep_len)
print(f"Masked random: score={np.mean(scores):.0f} max_tile mean={np.mean(tiles):.1f} len={np.mean(lengths):.0f}")

# %% [markdown]
# ## 随机基线（无 mask）：完全随机选动作

# %%
scores2, tiles2, lengths2 = [], [], []
for _ in range(200):
    env.reset()
    ep_len = 0
    while True:
        _, _, done = env.step(np.random.randint(4))
        ep_len += 1
        if done or ep_len >= 5000:
            break
    scores2.append(env.get_score())
    tiles2.append(env.get_max_tile())
    lengths2.append(ep_len)
print(f"Unmasked random: score={np.mean(scores2):.0f} max_tile={np.mean(tiles2):.1f} len={np.mean(lengths2):.0f}")

# %% [markdown]
# ## 单向基线：4 个方向各只按一个方向

# %%
for dir_idx, dir_name in enumerate(["up", "down", "left", "right"]):
    dir_scores = []
    for _ in range(200):
        env.reset()
        for __ in range(5000):
            _, _, done = env.step(dir_idx)
            if done:
                break
        dir_scores.append(env.get_score())
    print(f"  {dir_name}: avg_score={np.mean(dir_scores):.0f}")
