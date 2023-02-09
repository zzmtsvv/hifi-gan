import numpy as np
import wandb
from datetime import datetime
import torch
from configs import train_config


class WandBWriter:
    def __init__(self, config=train_config()) -> None:
        self.writer = None
        self.selected_module = ""

        wandb.login()

        if not hasattr(config, "wandb_project"):
            raise ValueError("please specify project name for wandb")
        
        wandb.init(
            project=getattr(config, 'wandb_project'),
            config=config
        )
        self.wb = wandb

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()
    
    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

        if not step:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()
    
    def scalar_name(self, scalar_name):
        return f"{self.mode}/{scalar_name}"
    
    def add_scalar(self, scalar_name, scalar):
        self.wb.log({
            self.scalar_name(scalar_name): scalar
        }, step=self.step)
    
    def add_scalars(self, tag, scalars):
        self.wb.log({
            **{f"{scalar_name}_{tag}_{self.mode}": scalar for scalar_name, scalar in scalars.items()}
        }, step=self.step)
    
    def add_image(self, scalar_name, image):
        self.wb.log({
            self.scalar_name(scalar_name): self.wb.Image(image)
        }, step=self.step)
    
    def add_audio(self, scalar_name, audio: torch.Tensor, sample_rate=None):
        audio = audio.detach().cpu().numpy().T
        self.wb.log({
            self.scalar_name(scalar_name): self.wb.Audio(audio, sample_rate=sample_rate)
        }, step=self.step)
    
    def add_text(self, scalar_name, text):
        self.wb.log({
            self.scalar_name(scalar_name): self.wb.Html(text)
        }, step=self.step)
    
    def add_histogram(self, scalar_name, hist: torch.Tensor, bins=None):
        hist = hist.detach().cpu().numpy()
        tmp = np.histogram(hist, bins=bins)

        if tmp[0].shape[0] > 512:
            tmp = np.histogram(hist, bins=512)
        
        hist = self.wb.Histogram(
            np_histogram=tmp
        )

        self.wb.log({
            self.scalar_name(scalar_name): hist
        }, step=self.step)
