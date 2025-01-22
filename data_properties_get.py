import numpy as np
from datasets import load_dataset
from logging import DEBUG
from flwr.common.logger import log

def compute_sft_trainer_steps(cfg, train_set):
    num_rounds = cfg.simulation.num_rounds
    total_steps = len(train_set)
    num_unique_rounds = int(np.ceil(num_rounds / cfg.simulation.eqivalent_cent_epochs))
    client_remainder = total_steps % cfg.simulation.num_clients
    client_data_size = total_steps // cfg.simulation.num_clients
    train_set = train_set.select(range(total_steps - client_remainder))

    client_round_data_size = client_data_size // num_unique_rounds
    cleint_ds_remainder = client_round_data_size % num_unique_rounds
    total_loss = cleint_ds_remainder * cfg.simulation.num_clients + client_remainder
    log(DEBUG,
        "Total centralised equivalnet data size: %s",
        client_round_data_size * num_rounds * cfg.simulation.num_clients,
    )
    log(DEBUG,"Total lossed elements: %s", total_loss)
    client_round_data_size = int(
        np.ceil(client_round_data_size / cfg.training.per_device_train_batch_size)
    )
    log(DEBUG,"Total expected steps per round: %s", client_round_data_size)
    return client_round_data_size

if __name__ == "__main__":
    import argparse
    import hydra
    from omegaconf import DictConfig

    parser = argparse.ArgumentParser(description="Compute SFT trainer steps")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument("train_data_path", type=str, help="Path to the training data file")

    args = parser.parse_args()

    @hydra.main(version_base=None, config_path=args.config_path, config_name="config")
    def main(cfg: DictConfig):
        train_set = load_dataset("json", data_files=args.train_data_path, split="train")
        compute_sft_trainer_steps(cfg, train_set)

    main()
