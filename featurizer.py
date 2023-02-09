from configs import MelSpectrogramConfig
import torch
from torch import nn
from librosa.filters import mel
from torchaudio import transforms


class MelSpectrogram(nn.Module):
    def __init__(self, config=MelSpectrogramConfig()) -> None:
        super().__init__()

        self.config = config

        self.mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels)
        self.mel_spectrogram.spectrogram.power = config.power

        mel_basis = mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.Tensor(mel_basis))
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spectrogram(audio).clamp(min=1e-5).log()
        return mel
