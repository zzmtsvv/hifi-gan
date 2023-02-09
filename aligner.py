from dataclasses import dataclass
import torchaudio
import torch
from torch import nn
from configs import MelSpectrogramConfig
from typing import Union, List
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


class GraphemeAligner(nn.Module):
    '''
        NOT WORKING
    '''
    def __init__(self) -> None:
        super().__init__()

        self.wav2vec2 = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        self.labels = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_labels()
        self.char2idx = {char: i for i, char in enumerate(self.labels)}
        self.unk_idx = self.char2idx['<unk>']
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=MelSpectrogramConfig.sr, new_freq=16000
        )
    
    def decode_text(self, text: str):
        text = text.replace(' ', '|').upper()
        return torch.tensor([
            self.char2idx.get(char, self.unk_idx) for char in text
        ]).long()
    
    def get_trellis(self, emission, tokens, blank_id=0):
        num_frames = emission.size(0)
        num_tokens = len(tokens)

        # Trellis has extra dimension for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.full((num_frames + 1, num_tokens + 1), -float('inf'))
        trellis[:, 0] = 0

        for t in range(num_frames):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        
        return trellis

    def backtrack(self, trellis, emission, tokens, blank_id=0):
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()
        path = []

        for t in range(t_start, 0, -1):
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
            prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()

            path.append(Point(j - 1, t - 1, prob))

            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("failed to align")
        
        return path[::-1]

    def merge_repeats(self):
        pass
    
    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor,
        wav_lengths: torch.Tensor,
        texts: Union[str, List[str]]
    ):
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = wavs.shape[0]
        durations = []

        for index in range(batch_size):
            current_wav = wavs[index, :wav_lengths[index]].unsqueeze(0)
            current_wav = self.resampler(current_wav)
            emission, _ = self.wav2vec2(current_wav)
            emission = emission.log_softmax(dim=-1).squeeze(0).cpu()

            tokens = self.decode_text(texts[index])

            trellis = self.get_trellis(emission, tokens)
            path = self.backtrack(trellis, emission, tokens)
            segments = self.merge_repeats(texts[index], path)

            num_frames = emission.shape[0]
            relative_durations = torch.tensor([
                segment.length / num_frames for segment in segments
            ])

            durations.append(relative_durations)
        
        durations = pad_sequence(durations).transpose(0, 1)
        return durations
    
    @staticmethod
    def plot_trellis_with_path(trellis, path):
        trellis_with_path = trellis.clone()
        
        for i, point in enumerate(path):
            trellis_with_path[point.time_index, point.token_index] = float('nan')
        plt.imshow(trellis_with_path[1:, 1:].T, origin='lower')
        plt.show()
