import torch
from . import featurizers
class ASR(torch.nn.Module):
    def __init__(self, configs: dict) -> None:
        super().__init__()
        self.configs = configs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.signal_featurizer = featurizers.LogMelSpectogram(configs)
        

    def forward(self, batch):
        ids, signals, trans, tokens = batch
        spectograms = self.signal_featurizer(signals)
        
        spectograms, tokens = spectograms.to(self.device), tokens.to(self.device)
        print(spectograms, tokens)