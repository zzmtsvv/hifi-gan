import torch
from torch import nn
from configs import train_config


class feature_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, features_real, features_generated):
        loss = 0

        for real, generated in zip(features_real, features_generated):
            loss += self.loss(real, generated)
        
        return 2 * loss


class discriminator_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.loss = nn.MSELoss()
    
    def forward(
        self,
        discriminator_real_outputs,
        discriminator_generated_outputs):

        loss = 0

        for real, generated in zip(discriminator_real_outputs, discriminator_generated_outputs):
            labels_true = torch.ones(real.size(), device=train_config.device_str)
            labels_fake = torch.zeros(generated.size(), device=train_config.device_str)

            loss += self.loss(real, labels_true) + self.loss(generated, labels_fake)
        
        return loss


class generator_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.MSELoss()
    
    def forward(self, discriminator_outputs):
        loss = 0

        for dg in discriminator_outputs:
            loss += self.loss(dg, torch.ones(dg.size(), device=train_config.device_str))
        
        return loss
