import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import wandb
import numpy as np
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.envs.wrappers import Game2048Wrapper
from src.agents.q_agent import DQNAgent
from src.buffers.replay import ReplayBuffer
from src.metrics.collector import MetricCollector
from src.engine.trainer import RLTrainer
from src.engine.evaluator import Evaluator


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()

    wandb_cfg = yaml.safe_load(open("configs/wandb.yaml"))
    wandb.init(project=wandb_cfg["project"], config=config)
    run_id = wandb.run.name

    env = Game2048Wrapper(penalty_invalid=config["env"]["penalty_invalid"])

    agent = DQNAgent(
        n_actions=env.n_actions,
        config=config["agent"],
        device=device,
    )
    buffer = ReplayBuffer(capacity=config["buffer"]["capacity"])
    collector = MetricCollector(use_wandb=True)

    trainer = RLTrainer(agent, buffer, config["train"]["batch_size"])
    evaluator = Evaluator(env, agent, collector, run_id, config["train"]["n_eval_episodes"])

    step = 0
    episode = 0
    ckpt_dir = Path(f"outputs/checkpoints/{run_id}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    while step < config["train"]["total_steps"]:
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        while True:
            valid_actions = np.array(env.get_valid_actions(), dtype=bool)
            action = agent.select_action(state, eval_mode=False, action_mask=valid_actions)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            step += 1
            episode_steps += 1

            if (step % config["train"]["train_frequency"] == 0
                    and len(buffer) >= config["train"]["batch_size"]):
                for _ in range(config["train"]["updates_per_step"]):
                    m = trainer.update()
                    if m:
                        collector.log(m, agent.train_step)

            if done or step >= config["train"]["total_steps"] \
                    or episode_steps >= config["train"]["max_episode_steps"]:
                break

        episode += 1
        collector.log({
            "episode/return": episode_reward,
            "episode/score": env.get_score(),
            "episode/max_tile": env.get_max_tile(),
            "episode/epsilon": agent.epsilon,
            **env.get_episode_stats(),
        }, step)

        if episode % config["train"]["eval_episode_interval"] == 0:
            evaluator.evaluate(step)
            ckpt = ckpt_dir / f"ep_{episode}.pt"
            agent.save(str(ckpt), agent_config=config["agent"])

    agent.save(str(ckpt_dir / "final.pt"), agent_config=config["agent"])
    wandb.finish()
