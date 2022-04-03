import os
import utils
import sentencepiece

if __name__ == "__main__":
    configs = utils.parsing.parse_args_and_configs()
    
    if not configs['skip_data_preparation']:
        utils.data.prepare_text_file(configs)
        
    if not os.path.exists(configs['result_dir']):
        os.makedirs(configs['result_dir'], exist_ok=True)
    
    tokenizer_prefix = f"{configs['model_type']}_{configs['vocab_size']}"
    tokenizer_prefix = os.path.join(configs['result_dir'], tokenizer_prefix)
    training_options = f"--input={configs['text_file']}\
        --model_prefix={tokenizer_prefix}\
        --vocab_size={configs['vocab_size']}\
        --model_type={configs['model_type']}\
        --character_coverage={configs['character_coverage']}"
    
    trainer = sentencepiece.SentencePieceTrainer
    trainer.train(training_options)