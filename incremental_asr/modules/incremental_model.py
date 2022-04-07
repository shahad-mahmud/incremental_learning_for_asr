import torch
import speechbrain as sb

from speechbrain.pretrained import Pretrained, fetching
from speechbrain.utils.distributed import run_on_main

from . import losses
from .teacher import Teacher


class ASR(sb.Brain):
    def __init__(  # noqa: C901
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
        teacher_dir=None,
    ):
        super().__init__(
            modules=modules,
            opt_class=opt_class,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
        )

        if teacher_dir is None:
            raise ValueError("teacher_dir must be specified")
        self.teacher = Teacher.from_hparams(teacher_dir,
                                            run_opts={'device': self.device})

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        features, self.feature_lengths = self.prepare_features(
            stage, batch.sig)
        tokens_bos, _ = self.prepare_tokens(stage, batch.tokens_bos)

        encoder_outs = self.modules.encoder(features.detach())

        embedded_tokens = self.modules.embedding(tokens_bos)
        decoder_outs, _ = self.modules.decoder(embedded_tokens, encoder_outs,
                                               self.feature_lengths)

        logits = self.modules.seq_lin(decoder_outs)
        predictions = {"seq_log_probs": self.hparams.log_softmax(logits)}
        predictions['teacher_log_probs'] = self.teacher.compute_probs(
            features, self.feature_lengths, tokens_bos)

        if self.is_ctc_active(stage):
            ctc_logits = self.modules.ctc_lin(encoder_outs)
            predictions["ctc_log_probs"] = self.hparams.log_softmax(ctc_logits)
        elif stage == sb.Stage.VALID:
            predictions["tokens"], _ = self.hparams.valid_search(
                encoder_outs, self.feature_lengths)
        elif stage == sb.Stage.TEST:
            predictions["tokens"], _ = self.hparams.valid_search(
                encoder_outs, self.feature_lengths)
        return predictions

    def compute_objectives(self, predictions, batch, stage):
        tokens_eos, tokens_eos_lens = self.prepare_tokens(
            stage, batch.tokens_eos)
        loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["seq_log_probs"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )

        if self.is_ctc_active(stage):
            tokens, tokens_lens = self.prepare_tokens(stage, batch.tokens)
            loss_ctc = self.hparams.ctc_cost(predictions["ctc_log_probs"],
                                             tokens, self.feature_lengths,
                                             tokens_lens)
            loss *= 1 - self.hparams.ctc_weight
            loss += self.hparams.ctc_weight * loss_ctc

        if stage != sb.Stage.TRAIN:
            predicted_words = [
                self.hparams.tokenizer.decode_ids(prediction).split(" ")
                for prediction in predictions["tokens"]
            ]
            target_words = [words.split(" ") for words in batch.words]

            self.wer_metric.append(batch.id, predicted_words, target_words)
            self.cer_metric.append(batch.id, predicted_words, target_words)

        return loss

    def prepare_features(self, stage, audio):
        audio, lengths = audio

        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(audio, lengths)
                audio = torch.cat([audio, wavs_noise], dim=0)
                lengths = torch.cat([lengths, lengths])

            if hasattr(self.hparams, "augmentation"):
                audio = self.hparams.augmentation(audio, lengths)

        features = self.hparams.compute_features(audio)
        features = self.modules.normalize(features, lengths)

        return features, lengths

    def prepare_tokens(self, stage, tokens):
        tokens, token_lens = tokens
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens], dim=0)
        return tokens, token_lens

    def is_ctc_active(self, stage):
        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.hparams.epoch_counter.current
        return current_epoch <= self.hparams.number_of_ctc_epochs

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        elif stage == sb.Stage.VALID:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current
                },
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
