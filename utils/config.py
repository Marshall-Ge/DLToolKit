import argparse
import os
import yaml

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __getattr__(self, item):
        return self.__dict__.get(item)

def update_config(config, args):
    #TODO: set local rank for distributed training
    pass

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    update_config(config, args)
    config = Config(config)
    return config

def parse_args(name):
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('--config', '-cfg', type=str, default="main.yaml")
    args = parser.parse_args()
    cfg = get_config(args)
    return cfg