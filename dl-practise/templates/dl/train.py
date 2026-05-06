import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import wandb
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.metrics.collector import MetricCollector
from src.engine.trainer import Trainer


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()

    wandb_config = yaml.safe_load(open("configs/wandb.yaml"))
    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config.get("entity"),
        config=config,
        tags=wandb_config.get("tags", []),
    )
    run_id = wandb.run.name

    collector = MetricCollector(use_wandb=True)
    trainer = Trainer(config, device, collector, run_id)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
