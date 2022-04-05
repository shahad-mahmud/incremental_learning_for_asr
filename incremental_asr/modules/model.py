from turtle import forward
import torch
import sentencepiece
from tqdm import tqdm
from . import featurizers
from . import blocks
from . import decoding
import utils


class ASR(torch.nn.Module):
    def __init__(self, configs: dict) -> None:
        super().__init__()
        self.configs = configs
        self.model = RnnASR(configs)

        self.log_prob = torch.nn.functional.log_softmax
        self.ctc = torch.nn.CTCLoss(blank=0, reduction='mean')
        self.__configure_optimizers()

        # for decoding
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=configs['tokenizer_path'])

    def forward(self, batch):
        _, loss = self.__step(batch)
        return loss

    def __step(self, batch):
        _, _, _, tokens, sig_lens, tok_lens = batch
        model_outputs = self.model(batch)
        probs = self.log_prob(model_outputs, dim=-1)

        probs = probs.transpose(0, 1)
        sig_lens = torch.full(size=(probs.shape[1], ),
                              fill_value=probs.shape[0],
                              dtype=torch.int32).to(probs.device)

        loss = self.ctc(probs, tokens, sig_lens, tok_lens)
        probs = probs.transpose(0, 1)
        return probs, loss

    def __configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=self.configs['lr_decay_factor'],
            verbose=True,
        )

    def fit(self, train_loader, valid_loader, epochs=5):
        for epoch in range(epochs):
            self.__fit_training_dataloader(train_loader, epoch + 1)
            self.__fit_validation_dataloader(valid_loader, epoch + 1)

    def __fit_validation_dataloader(self, valid_loader, epoch_num):
        average_loss = 0.0
        average_wer = 0.0
        self.model.eval()
        with tqdm(
                valid_loader,
                desc=f'Validation',
                dynamic_ncols=True,
        ) as loader:
            for batch in loader:
                probabilities, loss = self.__step(batch)

                hypos = decoding.greedy_decode(probabilities, self.tokenizer)
                wer = utils.scoring.calculate_batch_wer(batch[2], hypos)

                average_wer += wer
                average_loss += loss.item()
                loader.set_postfix(loss=average_loss / (loader.n + 1),
                                   wer=average_wer / (loader.n + 1))

    def __fit_training_dataloader(self, train_loader, epoch_num):
        average_loss = 0.0
        self.model.train()
        with tqdm(
                train_loader,
                desc=f'Epoch {epoch_num}',
                dynamic_ncols=True,
        ) as loader:
            for batch in loader:
                self.optimizer.zero_grad()
                _, loss = self.__step(batch)
                loss.backward()
                self.optimizer.step()

                average_loss += loss.item()
                loader.set_postfix(loss=average_loss / (loader.n + 1))


class RnnASR(torch.nn.Module):
    def __init__(self, configs: dict) -> None:
        super().__init__()
        self.configs = configs
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.signal_featurizer = featurizers.LogMelSpectogram(configs).to(
            self.device)

        self.conv = blocks.Conv1dBlock(
            in_channels=configs['n_mels'],
            out_channels=configs['n_mels'],
            kernel_size=configs['conv_kernel_size'],
            stride=configs['conv_stride'],
            padding=configs['conv_padding'],
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(configs['n_mels'], 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Dropout(p=configs['dropout']),
            torch.nn.Linear(128, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Dropout(p=configs['dropout']),
        )

        self.rnn = torch.nn.LSTM(input_size=256,
                                 hidden_size=512,
                                 num_layers=4,
                                 dropout=configs['dropout'],
                                 bidirectional=True,
                                 batch_first=True)

        self.layer_norm = torch.nn.LayerNorm(512 * 2)
        self.dropout = torch.nn.Dropout(p=configs['dropout'])
        self.linear = torch.nn.Linear(512 * 2, configs['dense_units'])

    def forward(self, batch):
        _, signals, _, _, _, _ = batch
        signals = signals.to(self.device)
        spectograms = self.signal_featurizer(signals)

        outputs = self.__compute_model_outputs(spectograms)
        return outputs

    def __compute_model_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2)
        outputs = self.conv(outputs)
        outputs = outputs.transpose(1, 2)

        outputs = self.dense(outputs)

        outputs, _ = self.rnn(outputs)

        outputs = self.layer_norm(outputs)
        outputs = torch.nn.functional.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear(outputs)

        return outputs


class AttnASR(torch.nn.Module):
    def __init__(self, configs: dict) -> None:
        super().__init__()
        self.configs = configs
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.signal_featurizer = featurizers.LogMelSpectogram(configs).to(
            self.device)

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
        signals = signals.to(self.device)
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