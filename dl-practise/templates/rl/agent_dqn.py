import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQNAgent:
    def __init__(self, n_actions: int, input_shape, config: dict = None, device=None):
        cfg = config or {}
        self.device = device or torch.device("cpu")
        self.n_actions = n_actions

        self.gamma = cfg.get("gamma", 0.99)
        self.lr = cfg.get("lr", 0.001)
        self.epsilon = cfg.get("epsilon_start", 1.0)
        self.epsilon_end = cfg.get("epsilon_end", 0.01)
        self.epsilon_decay = cfg.get("epsilon_decay", 50000)
        self.target_update = cfg.get("target_update", 1000)
        self.grad_clip_norm = cfg.get("grad_clip_norm", 10.0)

        model_cfg = cfg.get("model", {})
        self.q_network = self._build_network(n_actions, input_shape, model_cfg).to(self.device)
        self.target_network = self._build_network(n_actions, input_shape, model_cfg).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.train_step = 0

    def _build_network(self, n_actions, input_shape, model_cfg):
        from .network_cnn import CNNQNetwork
        return CNNQNetwork(
            n_actions=n_actions,
            input_shape=input_shape,
            conv_channels=model_cfg.get("conv_channels"),
            conv_kernels=model_cfg.get("conv_kernels"),
            fc_hidden=model_cfg.get("fc_hidden", 256),
        )

    def select_action(self, state, eval_mode=False, action_mask=None):
        if not eval_mode and np.random.random() < self.epsilon:
            if action_mask is not None and action_mask.any():
                valid = np.where(action_mask)[0]
                return np.random.choice(valid)
            return np.random.randint(self.n_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t).squeeze(0)

        if action_mask is not None:
            q_values[~torch.tensor(action_mask, device=self.device)] = float("-inf")

        return q_values.argmax().item()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(dim=1)[0]
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self._decay_epsilon()
        self.train_step += 1

        if self.train_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"train/loss": loss.item(), "episode/epsilon": self.epsilon}

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay)

    def save(self, path, agent_config=None):
        data = {"model_state": self.q_network.state_dict(), "train_step": self.train_step, "epsilon": self.epsilon}
        if agent_config:
            data["agent_config"] = agent_config
        torch.save(data, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(ckpt["model_state"])
        self.target_network.load_state_dict(ckpt["model_state"])
        self.train_step = ckpt.get("train_step", 0)
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
