import os
import utils
import torch
import modules
import sentencepiece
import pytorch_lightning as pl
from tqdm import tqdm

if __name__ == "__main__":
    configs = utils.parsing.parse_args_and_configs()

    if not configs['skip_data_preparation']:
        utils.data.prepare_annotation_files(configs)

    if not os.path.exists(configs['result_dir']):
        os.makedirs(configs['result_dir'], exist_ok=True)

    tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=configs['tokenizer_path'])

    train_loader = modules.data.SpeechDataLoader('train', configs, tokenizer)
    valid_loader = modules.data.SpeechDataLoader('valid', configs, tokenizer)
    test_loader = modules.data.SpeechDataLoader('test', configs, tokenizer)

    model = modules.model.ASR(configs)
    trainer = pl.Trainer(
        max_epochs=configs['epochs'],
        accelerator='gpu',
        devices=1
    )
    
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        # val_dataloaders=valid_loader,
    )
