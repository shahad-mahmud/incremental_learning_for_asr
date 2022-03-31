import yaml
import argparse


def parse_args_and_configs() -> dict:
    """Parse command line arguments and config file and returns the configs

    Returns:
        dict: Configs from the config file and command line arguments
    """
    parser = argparse.ArgumentParser(description='ASR arguments')
    parser.add_argument('config_file', type=str, help='Path to the YAML config file')
    parser.add_argument('--batch_size', type=int, help='batch size')

    args = vars(parser.parse_args())
    configs = parse_configs(args['config_file'])
    del args['config_file']

    for key, value in args.items():
        if value is not None:
            configs[key] = value

    return configs


def parse_configs(config_file_path):
    with open(config_file_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs
