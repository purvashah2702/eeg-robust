import hydra
from omegaconf import DictConfig
from bci.utils.seed import set_seed

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    datamodule = hydra.utils.instantiate(cfg.data)
    model      = hydra.utils.instantiate(cfg.model)
    trainer    = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train()
