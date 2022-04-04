import torch
import pytorch_lightning as pl
from . import featurizers
from . import blocks


class ASR(pl.LightningModule):
    def __init__(self, configs: dict) -> None:
        super().__init__()
        self.configs = configs
        self.model = ASRBase(configs)

        self.log_prob = torch.nn.functional.log_softmax
        self.ctc = torch.nn.CTCLoss(blank=0, reduction='mean')

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch):
        _, _, _, tokens, sig_lens, tok_lens = batch
        model_outputs = self.forward(batch)
        probs = self.log_prob(model_outputs, dim=-1)

        probs = probs.transpose(0, 1)
        sig_lens = torch.full(
            size=(probs.shape[1],),
            fill_value=probs.shape[0],
            dtype=torch.int32).to(probs.device)

        loss = self.ctc(probs, tokens, sig_lens, tok_lens)
        return loss

    def validation_step(self, batch, _):
        _, _, _, tokens, sig_lens, tok_lens = batch
        model_outputs = self.forward(batch)
        probs = self.log_prob(model_outputs, dim=-1)

        probs = probs.transpose(0, 1)
        sig_lens = torch.full(
            size=(probs.shape[1],),
            fill_value=probs.shape[0],
            dtype=torch.int32).to(probs.device)

        loss = self.ctc(probs, tokens, sig_lens, tok_lens)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
            lr=self.configs['learning_rate'])
        return optimizer


class ASRBase(torch.nn.Module):
    def __init__(self, configs: dict) -> None:
        super().__init__()
        self.configs = configs
        self.signal_featurizer = featurizers.LogMelSpectogram(configs)

        self.conv1 = blocks.Conv1dBlock(
            in_channels=configs['n_mels'],
            out_channels=configs['n_mels'],
            kernel_size=configs['conv_kernel_size'],
            stride=configs['conv_stride'],
            padding=configs['conv_padding'],
        )
        
        self.conv2 = blocks.Conv1dBlock(
            in_channels=configs['n_mels'],
            out_channels=configs['n_mels'],
            kernel_size=configs['conv_kernel_size'],
            stride=configs['conv_stride'],
            padding=configs['conv_padding'],
        )

        self.sab1 = blocks.AttentionBlock(
            configs['n_mels'],
            configs['attention_heads'],
            dropout=configs['dropout'],
        )
        self.sab2 = blocks.AttentionBlock(
            configs['n_mels'],
            configs['attention_heads'],
            dropout=configs['dropout'],
        )

        self.dense1 = torch.nn.Linear(
            configs['n_mels'],
            512,
        )
        self.dense2 = torch.nn.Linear(
            512,
            configs['dense_units'],
        )
        self.activation = torch.nn.LeakyReLU()
        self.norm = torch.nn.LayerNorm(configs['n_mels'])

    def forward(self, batch):
        _, signals, _, _, _, _ = batch
        spectograms = self.signal_featurizer(signals)

        outputs = self.__compute_model_outputs(spectograms)
        return outputs

    def __compute_model_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2)

        outputs = self.sab1(outputs)
        outputs = self.sab2(outputs)

        outputs = self.dense1(outputs)
        outputs = self.activation(outputs)
        outputs = self.dense2(outputs)
        outputs = self.activation(outputs)

        return outputs