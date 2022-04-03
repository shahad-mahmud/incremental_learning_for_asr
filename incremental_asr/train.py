import utils
import torch
import modules

if __name__ == "__main__":
    configs = utils.parsing.parse_args_and_configs()
    
    if not configs['skip_data_preparation']:
        utils.data.prepare_annotation_files(configs)
    
    train_set = modules.data.SpeechDataset(configs['train_annotation'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=configs['batch_size'], shuffle=True)
    
    test_set = modules.data.SpeechDataset(configs['test_annotation'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=configs['batch_size'], shuffle=False)
    
    valid_set = modules.data.SpeechDataset(configs['valid_annotation'])
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=configs['batch_size'], shuffle=False)
    
    
    
    