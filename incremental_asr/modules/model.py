import torch

class ASR(torch.nn.Module):
    def __init__(self, configs: dict) -> None:
        super().__init__()
        self.configs = configs
        

    def forward(self, batch):
        print(batch)