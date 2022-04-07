import os
import time
import torch
import speechbrain as sb

from tqdm import tqdm
from enum import Enum, auto
from torch.utils.data import DataLoader

from speechbrain.pretrained import Pretrained, fetching
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import LoopedLoader

from . import losses
from .teacher import Teacher


class ASR(sb.Brain):
    def __init__(  
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
        
        teacher_model_path = os.path.join(teacher_dir, 'save', 'model.ckpt')
        teacher_state_dicts = torch.load(teacher_model_path)
        self.hparams.model.load_state_dict(teacher_state_dicts)
        
        self.avg_rbkd_loss = 0

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
        predictions = {"student_log_probs": self.hparams.log_softmax(logits)}
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
        student_loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["student_log_probs"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )
        
        rbkd_loss = losses.rbkd(predictions["teacher_log_probs"], predictions["student_log_probs"])
        loss = student_loss + self.hparams.rbkd_factor * rbkd_loss
        
        self.avg_rbkd_loss = self.update_average(
            rbkd_loss, self.avg_rbkd_loss
        )

        if self.is_ctc_active(stage):
            tokens, tokens_lens = self.prepare_tokens(stage, batch.tokens)
            ctc_loss = self.hparams.ctc_cost(predictions["ctc_log_probs"],
                                             tokens, self.feature_lengths,
                                             tokens_lens)
            loss *= 1 - self.hparams.ctc_weight
            loss += self.hparams.ctc_weight * ctc_loss

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
    
    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        for epoch in epoch_counter:

            # Training stage
            self.on_stage_start(Stage.TRAIN, epoch)
            self.modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
            ) as t:
                for batch in t:
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
                    t.set_postfix(train_loss=self.avg_train_loss, rbkd_loss=self.avg_rbkd_loss)

                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        run_on_main(self._save_intra_epoch_ckpt)
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[Stage.VALID, avg_valid_loss, epoch],
                    )

            # Debug mode only runs a few epochs
            if self.debug and epoch == self.debug_epochs:
                break
            
class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()
