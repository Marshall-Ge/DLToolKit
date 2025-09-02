import hydra
from dltoolkit.utils.utils import *

@hydra.main(config_path="config", config_name="img_cls", version_base=None)
def main(config) -> None:

    run_img_cls(config)


def run_img_cls(config) -> None:
    # configure strategy
    strategy = get_strategy(config)
    strategy.setup_distributed()

    # configure model
    # TODO: Support customize model now, plan to support hf models
    model = get_local_model(config, 'img_cls')
    strategy.print(model)



if __name__ == "__main__":
    main()