import hydra
from dltoolkit.utils.utils import *
import logging
logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="kl", version_base=None)
def main(config) -> None:
    run_img_cls(config)


def run_img_cls(config) -> None:
    # configure strategy
    strategy = get_strategy(config)
    strategy.setup_distributed()
    strategy.print(f"Configs: {config}")

    model_org = get_local_or_pretrained_model(config, 'img_cls')

    # start of temporal experiment code TODO: a more elegant way to do this
    model_class = model_org.__class__
    config_1 = config.copy()
    config_1.model.param = config.model_param1
    model_1 = model_class(config_1)
    config_2 = config.copy()
    config_2.model.param = config.model_param2
    model_2 = model_class(config_2)
    # end of experiment code


    kl_results, kl_mean = compute_model_kl_divergence(
        model_dir1 = config.model_dir1,
        model_dir2 = config.model_dir2,
        model_1 = model_1,
        model_2 = model_2,
        use_kde= True,
        skipped_layers=config.skipped_layers
    )

    strategy.print(f"kl_mean: {kl_mean}")


if __name__ == "__main__":
    main()