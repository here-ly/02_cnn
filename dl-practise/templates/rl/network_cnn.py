import torch.nn as nn


class CNNQNetwork(nn.Module):
    def __init__(self, n_actions, input_shape, conv_channels=None, conv_kernels=None, fc_hidden=256):
        super().__init__()
        conv_channels = conv_channels or [128, 128]
        conv_kernels = conv_kernels or [2, 2]

        layers = []
        in_ch = input_shape[0]  # channels first
        for out_ch, k in zip(conv_channels, conv_kernels):
            layers.append(nn.Conv2d(in_ch, out_ch, k))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(self._calc_conv_out(input_shape[1:]), fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_actions),
        )

    def _calc_conv_out(self, spatial):
        """自动计算 conv 输出的展平维度，避免硬编码全连接层输入 size。"""
        import torch
        dummy = torch.zeros(1, 1, *spatial)
        with torch.no_grad():
            # 构造临时 Sequential 只跑 conv 部分
            return self.features(dummy).view(1, -1).size(1)

    def forward(self, x):
        return self.head(self.features(x))
