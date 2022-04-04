import torch
from . import featurizers


class ASR(torch.nn.Module):
    def __init__(self, configs: dict) -> None:
        super().__init__()
        self.configs = configs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.signal_featurizer = featurizers.LogMelSpectogram(configs)

        # model components
        self.conv = []
        for i in range(configs['n_conv']):
            self.conv.append(
                torch.nn.Conv1d(
                    in_channels=configs['n_mels'],
                    out_channels=configs['n_mels'],
                    kernel_size=configs['conv_kernel_size'],
                    stride=configs['conv_stride'],
                    padding=configs['conv_padding'],
                    device=self.device,
                ))

        self.sabs = []
        for i in range(configs['n_sab']):
            self.sabs.append(
                torch.nn.MultiheadAttention(
                    configs['n_mels'],
                    configs['attention_heads'],
                    batch_first=True,
                    device=self.device,
                ))

        self.dense1 = torch.nn.Linear(configs['n_mels'],
                                      512,
                                      device=self.device)
        self.dense2 = torch.nn.Linear(512,
                                      configs['dense_units'],
                                      device=self.device)
        self.softmax = torch.nn.functional.log_softmax

        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean')

    def forward(self, batch):
        ids, signals, trans, tokens, sig_lens, tok_lens = batch
        spectograms = self.signal_featurizer(signals)

        spectograms, tokens = spectograms.to(self.device), tokens.to(
            self.device)
        outputs = self.__compute_model_outputs(spectograms)
        probalities = self.softmax(outputs)

        # get loss
        probalities = probalities.transpose(0, 1)
        sig_lens = torch.full(size=(probalities.shape[1], ),
                              fill_value=probalities.shape[0],
                              dtype=torch.int32,
                              device=self.device)
        loss = self.ctc_loss(probalities, tokens, sig_lens, tok_lens)
        return loss

    def __compute_model_outputs(self, inputs):
        outputs = inputs.transpose(1, 2)
        for conv in self.conv:
            outputs = conv(outputs)
        outputs = outputs.transpose(1, 2)

        for sab in self.sabs:
            outputs = sab(outputs, outputs, outputs)[0]

        outputs = self.dense1(outputs)
        outputs = self.dense2(outputs)

        return outputs