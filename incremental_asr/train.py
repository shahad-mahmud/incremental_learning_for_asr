import utils

if __name__ == "__main__":
    configs = utils.parsing.parse_args_and_configs()
    print(configs)
    utils.data.prepare_annotation_files(configs)
    
    