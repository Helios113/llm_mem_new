import numpy as np
from datasets import load_dataset
from logging import DEBUG
from flwr.common.logger import log

def compute_sft_trainer_steps(cfg, train_set_path):
    num_rounds = cfg.simulation.num_rounds
    train_set = load_dataset("json", data_files=train_set_path, split="train")
    total_steps = len(train_set)
    num_unique_rounds = int(np.ceil(num_rounds / cfg.simulation.eqivalent_cent_epochs))
    print("Rounds per epoch: ", num_unique_rounds)
    print("Epochs: ", cfg.simulation.eqivalent_cent_epochs)
    client_remainder = total_steps % cfg.simulation.num_clients
    client_data_size = total_steps // cfg.simulation.num_clients
    train_set = train_set.select(range(total_steps - client_remainder))

    client_round_data_size = client_data_size // num_unique_rounds
    cleint_ds_remainder = client_round_data_size % num_unique_rounds
    total_loss = cleint_ds_remainder * cfg.simulation.num_clients + client_remainder
    print(
        "Single Epoch Steps:",
        client_round_data_size * num_unique_rounds * cfg.simulation.num_clients,
    )
    print("Total lossed elements:", total_loss)

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    train_data_path = "/nfs-share/pa511/new_work/data/triviaqa/data_train.json"
    @hydra.main(version_base=None, config_path="/nfs-share/pa511/new_work/config/", config_name="config")
    def main(cfg: DictConfig):
        compute_sft_trainer_steps(cfg, train_data_path)

    main()
