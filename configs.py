from dataclasses import dataclass
import torch


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    pad_value: float = -11.5129251


@dataclass
class train_config:
    random_seed: int = 42
    device_str: str = 'cuda' if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dataset_directory = 'LJSpeech-1.1'
    num_epochs = 1000
    batch_size = 4
    lambda_ = 45
    discriminator_lr = 3e-4
    generator_lr = 3e-4
    validation_transcripts = [
        'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
        'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
        'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space',
        ]
    weights_path = ['best_msd.pt', 'best_mpd.pt', 'best_generator.pt']
    wandb_project = "HIFI-GAN"
    checkpoint_period = 10


@dataclass
class generator_config:
    first_hidden_size: int = 256
    upsample_kernel_sizes = (16, 16, 8)
    upsample_rates = (8, 8, 4)
    resblock_kernel_sizes = (3, 5, 7)
    resblock_dilation_sizes = ((1, 2), (2, 6), (3, 12))
    lr = 2e-4
    betas=(.8, .99)
    T_max = 2000
    eta_min = 1e-7
