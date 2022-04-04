import torch
import torchaudio

class LogMelSpectogram(torch.nn.Module):
    def __init__(self, configs: dict) -> None:
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mel_spectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=configs['sample_rate'],
            n_fft=configs['n_fft'],
            n_mels=configs['n_mels'],
            win_length=configs['win_length'],
            hop_length=configs['hop_length'],
        )
    
    def forward(self, input):
        input = input.to(self.device)
        mel_spectogram = self.mel_spectogram(input)
        log_mel_spectogram = torch.log(mel_spectogram + 1e-6)
        
        return log_mel_spectogram.transpose(1, 2)