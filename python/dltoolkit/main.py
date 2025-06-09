import yaml
import subprocess
from dltoolkit.configs.defaults import assert_and_infer_cfg
from dltoolkit.utils.parse import parse_args, load_config
import dltoolkit.tools as tools
from dltoolkit.utils.misc import launch_job


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    train = tools.train
    test = tools.test

    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)
        # Perform training.
        if cfg.TRAIN.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=train)
        # Perform testing.
        if cfg.TEST.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == '__main__':
    main()