import os
import torch
from hyperpyyaml import load_hyperpyyaml

from speechbrain.pretrained import Pretrained, fetching
from speechbrain.utils.distributed import run_on_main


class Teacher(Pretrained):
    MODULES_NEEDED = [
        "encoder",
        "decoder",
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_probs(self, features, feature_lengths, tokens):
        with torch.no_grad():
            encoder_out = self.modules.encoder(features.detach())
            
            embeddings = self.modules.embedding(tokens)
            decoder_outs, _ = self.modules.decoder(embeddings, encoder_out, feature_lengths)
            
            logits = self.modules.seq_lin(decoder_outs)
            
        return self.hparams.log_softmax(logits)    

    @classmethod
    def from_hparams(
        cls,
        source,
        hparams_file="hyperparams.yaml",
        overrides={},
        save_dir=None,
        use_auth_token=False,
        **kwargs,
    ):
        if save_dir is None:
            save_dir = os.path.join(source, 'save')

        hparams_path = fetching.fetch(hparams_file, source, save_dir,
                                      use_auth_token)

        with open(hparams_path) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        pretrainer = hparams["pretrainer"]
        pretrainer.add_loadables({'model': hparams['model']})

        pretrainer.set_collect_in(save_dir)
        run_on_main(pretrainer.collect_files,
                    kwargs={"default_source": source})
        pretrainer.load_collected(device="cpu")

        return cls(hparams["modules"], hparams, **kwargs)
