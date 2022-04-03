import utils
import torch
import modules

if __name__ == "__main__":
    configs = utils.parsing.parse_args_and_configs()
    
    if not configs['skip_data_preparation']:
        utils.data.prepare_annotation_files(configs)
    
    train_set = modules.data.SpeechDataset(configs['train_annotation'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=configs['batch_size'], shuffle=True)
    
    print(len(train_set))
    for a in train_set:
        print(a)
    
    # for batch in train_loader:
    #     print(batch)
    
    
    