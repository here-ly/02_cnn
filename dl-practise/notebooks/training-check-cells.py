"""
03_debug_training.py — 训练调试 notebook 的必须检查 cell。

用于验证网络前向、update、epsilon 衰减、checkpoint 等核心逻辑是否正确。
"""

# %% [markdown]
# ## 1. 网络前向：确认 Q-network input/output shape

# %%
dummy_state = np.random.randn(1, 4, 4).astype(np.float32)
state_t = torch.FloatTensor(dummy_state).unsqueeze(0).to(device)
q_values = agent.q_network(state_t)
print(f"Q-values shape: {q_values.shape}, values: {q_values.detach().cpu().numpy()}")

# %% [markdown]
# ## 2. 单次 update：验证 loss 不是 NaN、backward 正常

# %%
# 填充 buffer
for _ in range(64):
    s = np.random.randn(1, 4, 4).astype(np.float32)
    a = np.random.randint(4)
    s2 = np.random.randn(1, 4, 4).astype(np.float32)
    r = np.random.randn()
    d = False
    buffer.push(s, a, r, s2, d)

batch = buffer.sample(32)
loss = trainer.update()  # or agent.update(batch)
print(f"Loss: {loss['train/loss']:.4f}")
assert not np.isnan(loss["train/loss"]), "Loss is NaN!"

# %% [markdown]
# ## 3. epsilon 衰减检查

# %%
agent.epsilon = 1.0
for i in range(0, 50001, 10000):
    print(f"step {i}: epsilon = {agent.epsilon:.4f}")
    for _ in range(10000):
        agent._decay_epsilon()

# %% [markdown]
# ## 4. 目标网络同步验证

# %%
sum_before = sum(p.sum().item() for p in agent.q_network.parameters())
agent.target_network.load_state_dict(agent.q_network.state_dict())
sum_after_target = sum(p.sum().item() for p in agent.target_network.parameters())
assert sum_before == sum_after_target, "target network params not synced!"
print(f"Params sum: before={sum_before:.4f}, target after sync={sum_after_target:.4f}")

# %% [markdown]
# ## 5. 保存/加载 checkpoint

# %%
import tempfile, os
tmpdir = tempfile.mkdtemp()
ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
agent.save(ckpt_path)
agent2 = DQNAgent(n_actions=4, device=device)
agent2.load(ckpt_path)
assert agent2.epsilon == agent.epsilon, "epsilon mismatch"
assert agent2.train_step == agent.train_step, "train_step mismatch"
print(f"Loaded OK: epsilon={agent2.epsilon}, train_step={agent2.train_step}")
