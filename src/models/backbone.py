import torch
from torch import nn
from torch.nn import functional as F


class SimpleCNN(nn.Module):
    """
    配置驱动的卷积神经网络。
    网络结构（conv_channels, conv_kernels, fc_hidden 等）全部从 config dict 读取，
    零硬编码。
    """

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or {}
        self.input_channels = cfg.get("input_channels", 3)
        self.input_size = cfg.get("input_size", 32)
        self.num_classes = cfg.get("num_classes", 10)
        conv_channels = cfg.get("conv_channels", [32, 64, 128])
        conv_kernels = cfg.get("conv_kernels", [3, 3, 3])
        conv_paddings = cfg.get("conv_paddings", [1, 1, 1])
        self.pool_size = cfg.get("pool_size", 2)
        self.fc_hidden = cfg.get("fc_hidden", 256)

        layers = []
        in_ch = self.input_channels
        for out_ch, k, p in zip(conv_channels, conv_kernels, conv_paddings):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(self.pool_size, self.pool_size))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        conv_out_size = self.input_size
        for _ in conv_channels:
            conv_out_size = (conv_out_size + 2 * p - (k - 1) - 1) // self.pool_size + 1
        conv_out_size = conv_out_size // (self.pool_size ** len(conv_channels)) if conv_kernels[0] == 3 and conv_paddings[0] == 1 else self._calc_conv_out()

        self._conv_out_dim = self._calc_conv_out()
        self.classifier = nn.Sequential(
            nn.Linear(self._conv_out_dim, self.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_hidden, self.num_classes),
        )

    def _calc_conv_out(self) -> int:
        dummy = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
        with torch.no_grad():
            out = self.features(dummy)
        return out.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(config: dict) -> SimpleCNN:
    model_cfg = config.get("model", {})
    return SimpleCNN(model_cfg)
