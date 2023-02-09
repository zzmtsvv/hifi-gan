import torch
from torch import nn
from torch.nn import functional as F
from losses import feature_loss


class PeriodSubDiscriminator(nn.Module):
    def __init__(self, period) -> None:
        super().__init__()

        self.period = period
        self.convs = nn.Sequential(*self.build_conv_layers())

        self.fc = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
    
    def build_conv_layers(self):
        dims = [1, 32, 128, 512, 1024]
        layers = []

        for i in range(len(dims) - 1):
            layer = nn.utils.weight_norm(nn.Conv2d(dims[i], dims[i + 1], (5, 1), (3, 1)))
            layers.append(layer)
        
        layers.append(nn.utils.weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))))

        return layers
    
    def forward(self, x: torch.Tensor):
        features = []

        if x.size(-1) % self.period:
            x = F.pad(x, (0, self.period - (x.size(-1) % self.period)))
        x = x.view(x.size(0), x.size(1) // self.period, self.period).unsqueeze(1)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x)
            features.append(x)
        x = self.fc(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, features


class MPD(nn.Module):
    '''
        Multi-Period Discriminator
    '''
    def __init__(self) -> None:
        super().__init__()

        self.sub_discriminators = nn.Sequential(
            *[PeriodSubDiscriminator(period) for period in [2, 3, 5, 7, 11]]
        )
        self.loss = feature_loss()

    def enable_grads(self):
        for p in self.parameters():
            p.requires_grad = True
    
    def load(self, filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(filename, map_location=device)
        self.load_state_dict(state_dict)
    
    def disable_grads(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, real: torch.Tensor, generated: torch.Tensor):
        real_ = []
        generated_ = []
        loss = 0

        for i, sub_disc in enumerate(self.sub_discriminators):
            real_x, real_feature = sub_disc(real)
            generated_x, generated_feature = sub_disc(generated)
            
            real_.append(real_x)
            generated_.append(generated_x)
            loss += self.loss(real_feature, generated_feature)
        
        return real_, generated_, loss

    def get_model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ScaleSubDiscriminator(nn.Module):
    def __init__(self, norm=nn.utils.weight_norm) -> None:
        super().__init__()

        self.convs = nn.Sequential(
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        )
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        features = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)

        return x, features


class MSD(nn.Module):
    '''
        Multi-Scale Discriminator
    '''
    def __init__(self) -> None:
        super().__init__()

        self.loss = feature_loss()
        self.discriminators = nn.ModuleList([
            ScaleSubDiscriminator(norm=nn.utils.spectral_norm),
            ScaleSubDiscriminator(),
            ScaleSubDiscriminator()
        ])

        self.pool = nn.AvgPool1d(4, 2, 2)
    
    def load(self, filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(filename, map_location=device)
        self.load_state_dict(state_dict)
    
    def enable_grads(self):
        for p in self.parameters():
            p.requires_grad = True
    
    def disable_grads(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, real, generated):
        real_ = []
        generated_ = []
        loss = 0

        for i, discriminator in enumerate(self.discriminators):
            if i:
                real = self.pool(real)
                generated = self.pool(generated)
            real_x, real_features = discriminator(real)
            generated_x, generated_features = discriminator(generated)
            real_.append(real_x)
            generated_.append(generated_x)

            loss += self.loss(real_features, generated_features)
        
        return real_, generated_, loss

    def get_model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
