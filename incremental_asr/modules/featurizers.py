import torch
import torchaudio

class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, configs: dict) -> None:
        super().__init__()
        win_length = int(round((configs['sample_rate'] / 1000.0) * configs['win_length']))
        hop_length = int(round((configs['sample_rate'] / 1000.0) * configs['hop_length']))
        self.mel_spectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=configs['sample_rate'],
            n_fft=configs['n_fft'],
            n_mels=configs['n_mels'],
            win_length=win_length,
            hop_length=hop_length,
            normalized=True
        )
    
    def forward(self, input):
        mel_spectogram = self.mel_spectogram(input)
        log_mel_spectogram = torch.log(mel_spectogram + 1e-14)
        
        return log_mel_spectogram.permute(0, 2, 1)

class LogMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        n_mels: int = 80,
        win_length: int = 25,
        hop_length: int = 10,
        log_offset: float = 1e-6,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = int(
            round((self.sample_rate / 1000.0) * win_length)
        )
        self.hop_length = int(
            round((self.sample_rate / 1000.0) * hop_length)
        )
        self.log_offset = log_offset
        self.window = torch.hann_window 

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=self.window
        ).to(inputs.device)(inputs)
        log_mel_spec = torch.log(mel_spec + self.log_offset)

        return log_mel_spec.permute(0, 2, 1)