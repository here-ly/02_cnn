class MetricCollector:
    """指标统一入口，对接 wandb。"""

    def __init__(self, use_wandb: bool = True):
        self.use_wandb = use_wandb

    def log(self, metrics: dict, step: int):
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def log_images(self, images: list, step: int, caption: str = "images"):
        if self.use_wandb:
            import wandb
            wandb.log({caption: [wandb.Image(img) for img in images]}, step=step)
