from configs import train_config, generator_config, MelSpectrogramConfig
from dataset import LJSpeechDataset, LJSpeechCollator
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from discriminator import MSD, MPD
from losses import discriminator_loss, generator_loss
from generator import Generator
from aligner import GraphemeAligner
from featurizer import MelSpectrogram
from itertools import chain
import os
from PIL import Image
from tqdm import tqdm
from writer import WandBWriter
from utils import plot_spectrogram_to_buf


class Trainer:
    def __init__(self) -> None:

        self.writer = WandBWriter()

        #self.aligner = GraphemeAligner().to(train_config.device)
        self.dataloader = DataLoader(LJSpeechDataset('.'), batch_size=train_config.batch_size, collate_fn=LJSpeechCollator())
        self.featurizer = MelSpectrogram()
        
        self.generator = Generator().to(train_config.device)
        self.msd = MSD().to(train_config.device)
        self.mpd = MPD().to(train_config.device)

        self.mpd_loss = discriminator_loss()
        self.msd_loss = discriminator_loss()
        self.mel_loss = nn.L1Loss()
        self.gen_mpd_loss = generator_loss()
        self.gen_msd_loss = generator_loss()

        self.criteion = nn.MSELoss()
        self.optim_g = torch.optim.AdamW(self.generator.parameters(), lr=generator_config.lr, betas=generator_config.betas, eps=1e-9)
        self.optim_d = torch.optim.AdamW(
            chain(self.mpd.parameters(), self.msd.parameters()),
            lr=generator_config.lr,
            betas=generator_config.betas,
            eps=1e-9
        )
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_g, generator_config.T_max, generator_config.eta_min)
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_d, generator_config.T_max, generator_config.eta_min)
        
        self.tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
    
    def fit(self):
        print(f"Training starts on {train_config.device_str} 🚀")

        for epoch in tqdm(range(1, train_config.num_epochs + 1)):
            self.generator.train()
            self.msd.train()
            self.mpd.train()

            epoch_generator_loss_log = []
            epoch_discriminator_loss_log = []

            for i, batch in enumerate(self.dataloader):
                mels = batch.mels.to(train_config.device)
                waveforms = batch.waveform.to(train_config.device)

                waveform_preds = self.generator(mels).squeeze(1)
                waveform_preds = waveform_preds[:, :batch.waveform.size(-1)]
                melspec_preds = self.featurizer(waveform_preds.cpu()).to(train_config.device)

                # discriminators step
                self.optim_d.zero_grad()

                mpd_r, mpd_g, _ = self.mpd(waveforms, waveform_preds.detach())
                mpd_loss = self.mpd_loss(mpd_r, mpd_g)

                msd_r, msd_g, _ = self.msd(waveforms, waveform_preds.detach())
                msd_loss = self.msd_loss(msd_r, msd_g)
                
                discriminator_loss = msd_loss + mpd_loss
                discriminator_loss.backward()
                self.optim_d.step()
                
                # generator step
                self.optim_g.zero_grad()

                mel_loss = train_config.lambda_ * self.mel_loss(melspec_preds, mels)
                
                self.msd.disable_grads()
                self.mpd.disable_grads()

                mpd_r, mpd_g, mpd_feature_loss = self.mpd(waveforms, waveform_preds)
                msd_r, msd_g, msd_feature_loss = self.msd(waveforms, waveform_preds)
                gen_loss_mpd = self.gen_mpd_loss(mpd_g)
                gen_loss_msd = self.gen_msd_loss(msd_g)

                gen_loss = gen_loss_mpd + gen_loss_msd + msd_feature_loss + mpd_feature_loss + mel_loss

                gen_loss.backward()
                self.optim_g.step()

                self.msd.enable_grads()
                self.mpd.enable_grads()

                epoch_generator_loss_log = np.append(epoch_generator_loss_log, gen_loss.item())
                epoch_discriminator_loss_log = np.append(epoch_discriminator_loss_log, discriminator_loss.item())

                if not (i % train_config.checkpoint_period):
                    self.add_statistics(epoch, i, epoch_generator_loss_log, epoch_discriminator_loss_log)
                
                self.scheduler_g.step()
                self.scheduler_d.step()
        
        self.save()
    
    def add_statistics(
            self,
            epoch: int,
            iteration: int,
            gen_loss: np.ndarray,
            discriminator_loss: np.ndarray):
        
        self.writer.set_step(epoch * len(self.dataloader) + iteration)
        self.writer.add_scalar("learning rate", self.scheduler_g.get_last_lr()[0])
        self.writer.add_scalar("Train generator loss", gen_loss[iteration - train_config.checkpoint_period + 1:].mean())
        self.writer.add_scalar("Train discriminator loss", discriminator_loss[iteration - train_config.checkpoint_period + 1:].mean())

    def validate(self):
        val_batch = (
            self.featurizer(torchaudio.load('audio_1.wav')[0]),
            self.featurizer(torchaudio.load('audio_2.wav')[0]),
            self.featurizer(torchaudio.load('audio_3.wav')[0]))
        
        with torch.no_grad():
            self.generator.eval()
            generated_waves = (self.generator(sample.to(train_config.device)) for sample in val_batch)

            for audio, t in zip(generated_waves, train_config.validation_transcripts):
                image = Image.open(plot_spectrogram_to_buf(audio))
                self.writer.add_audio(f"Generated audio for '{t}'", audio, MelSpectrogramConfig.sr)
    
    def save(self):
        filenames = train_config.weights_paths
        directory = "weights"
        if not os.path.isdir(directory):
            os.makedirs(directory)
        
        torch.save(self.msd.state_dict(), filenames[0])
        torch.save(self.mpd.state_dict(), filenames[1])
        torch.save(self.generator.state_dict(), filenames[2])
