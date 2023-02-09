import io
import matplotlib.pyplot as plt
import torch
import random
import os
import numpy as np
from configs import train_config


def print_device_info(device=None):
    device =  torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using device: {device}')

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory usage:')
        print(f'Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB')
        print(f'Cached:    {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')


def plot_spectrogram_to_buf(reconstructed_wav, name=None):
    plt.figure(figsize=(20, 5))
    plt.plot(reconstructed_wav, alpha=.5)
    plt.title(name)
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def seed_everything(seed: int = train_config.random_seed) -> None:
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def disable_grads(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def enable_grads(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = True
