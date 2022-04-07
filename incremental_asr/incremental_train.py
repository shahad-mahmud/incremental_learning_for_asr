import os
import sys
import utils
import modules
import speechbrain as sb

from hyperpyyaml import load_hyperpyyaml

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    sb.utils.distributed.ddp_init_group(run_opts)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    if hparams['run_num'] == hparams['teacher_run_num']:
        raise ValueError("run_num and parent_run_num must be different")
        
    sb.create_experiment_directory(
        experiment_directory=hparams["result_dir"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    datasets = utils.sb.dataio_prepare(hparams)
    model = modules.incremental_model.ASR(
        modules=hparams['modules'],
        opt_class=hparams['opt_class'],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams['checkpointer'],
        teacher_dir=hparams['teacher_dir']
    )
    
    model.fit(
        epoch_counter=model.hparams.epoch_counter,
        train_set=datasets['train'],
        valid_set=datasets['valid'],
        train_loader_kwargs=hparams['train_dataloader_opts'],
        valid_loader_kwargs=hparams['valid_dataloader_opts'],
    )
    
    model.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )