import torch
from torch import nn
from torch.nn import functional as F
from configs import generator_config


class ConvBlock1d(nn.Module):
    def __init__(self, hidden_size, kernel_size, dilation) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size,
                stride=1,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size,
                stride=1,
                dilation=1,
                padding=(kernel_size - 1) // 2
            )
        )
    
    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResidualBlock1d(nn.Module):
    def __init__(self, hidden_size, kernel_size=3, dilations=(1, 3, 5)) -> None:
        super().__init__()

        self.block = nn.ModuleList(
            [ConvBlock1d(hidden_size, kernel_size, d) for d in dilations]
        )
    
    def forward(self, x: torch.Tensor):
        for layer in self.block:
            x = layer(x) + x
        
        return x


class MRF(nn.Module):
    '''
        Multi-Receptive Field
    '''
    def __init__(self, hidden_size, kernel_sizes, dilations) -> None:
        super().__init__()

        self.resblocks = nn.ModuleList([
            ResidualBlock1d(hidden_size, kernel_size, dilations_)
            for kernel_size, dilations_ in zip(kernel_sizes, dilations)
        ])

    def load(self, filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(filename, map_location=device)
        self.load_state_dict(state_dict)
    
    def forward(self, x: torch.Tensor):
        residual = None

        for resblock in self.resblocks:
            if residual is None:
                residual = resblock(x)
            else:
                residual += resblock(x)
        
        return residual / len(self.resblocks)


class Generator(nn.Module):
    def __init__(self, config=generator_config()) -> None:
        super().__init__()

        self.config = config
        self.num_upsamples = len(config.upsample_rates)

        self.conv_pre = nn.Conv1d(80, config.first_hidden_size, 7, 1, padding=3)
        self.convs = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    config.first_hidden_size // 2 ** i,
                    config.first_hidden_size // 2 ** (i + 1),
                    kernel_size=upblock_kernel_size,
                    stride=upblock_stride,
                    padding=(upblock_kernel_size - upblock_stride) // 2,
                )
                for i, (upblock_kernel_size, upblock_stride) in enumerate(zip(
                config.upsample_kernel_sizes, config.upsample_rates
            ))
            ]
        )

        self.MRFs = nn.ModuleList(
            [
                MRF(
                    config.first_hidden_size // 2 ** (i + 1),
                    config.resblock_kernel_sizes,
                    config.resblock_dilation_sizes,
                )
                for i in range(self.num_upsamples)
            ]
        )

        self.fc = nn.Conv1d(
            config.first_hidden_size // 2 ** self.num_upsamples,
            1, kernel_size=7, stride=1, padding=3
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_pre(x)

        for conv, mrf in zip(self.convs, self.MRFs):
            x = F.leaky_relu(x)
            x = conv(x)
            x = mrf(x)
        x = F.leaky_relu(x)
        x = self.fc(x)

        return torch.tanh(x)

    def load(self, filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(filename, map_location=device)
        self.load_state_dict(state_dict)

    def get_model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
