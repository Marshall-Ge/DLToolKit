import hydra
from dltoolkit.utils.utils import *
from accelerate import Accelerator

@hydra.main(config_path="config", config_name="itm_config", version_base=None)
def main(config) -> None:

    run_itm(config)


def run_itm(config) -> None:
    # configure strategy
    strategy = get_strategy(config)
    strategy.setup_distributed()

    # configure model
    # TODO: Support customize model now, plan to support hf models
    model = get_local_model(config, 'itm')






if __name__ == "__main__":
    main()