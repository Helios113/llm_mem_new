# Flower client imports
from flwr.client import NumPyClient
from flwr.client import ClientApp

# Flower server imports
from flwr.server.server_app import ServerApp
from flwr.server.server_config import ServerConfig
from flwr.server.serverapp_components import ServerAppComponents
from flwr.server.strategy import FedAvg

# Flower other imports
from flwr.simulation import run_simulation
from flwr.common import Context
from flwr.common.logger import log

# Flower datasets
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# Huggingface imports
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Generic imports
import logging
import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, open_dict
import csv
import random

# Our packages
from client import HuggingFaceClient
from data_formatting import get_tokenizer_and_data_collator_and_prompt_formatting
from client_manager import RandomOrgClientManager
from evalaute_server import get_evaluate_fn


# Fix random seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def fit_config_fn(server_round: int):
    fit_config = {"current_round": server_round}
    return fit_config


@hydra.main(version_base=None, config_path="config", config_name="config")
def start_flower_simulation(cfg: DictConfig):

    client_manager = RandomOrgClientManager("2e4c64c1-80ab-42b9-8924-9576140f571e")
    tokenizer, collator = get_tokenizer_and_data_collator_and_prompt_formatting(
        cfg.model.name, cfg.model.tokenizer, cfg.model.instruction_token
    )

    partitioner = IidPartitioner(num_partitions=cfg.simulation.num_clients)
    # Let's partition the "train" split of the MNIST dataset
    # The MNIST dataset will be downloaded if it hasn't been already

    fds = FederatedDataset(
        dataset="json",
        partitioners={"train": partitioner},
        data_files=cfg.train_dataset.files,
    )
    with open_dict(cfg):
        cfg.run_id = wandb.util.generate_id()
        cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Append run_id and output_dir to a table at a predefined path
    table_path = "/nfs-share/pa511/llm_memorisation/new_work/table.csv"
    with open(table_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cfg.run_id, cfg.output_dir])

    eval_set = load_dataset(
        "json",
        data_files=cfg.validation_dataset.files,
        split=cfg.validation_dataset.split,
    )

    # Create LoraConfig
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
    )

    # Client app initialisation
    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        print(f"Client {partition_id} started")
        train_data = fds.load_partition(partition_id, "train")

        # Split train_data into equal sections
        num_rounds = cfg.simulation.num_rounds
        total_steps = len(train_data)
        num_unique_rounds = int(np.ceil(
            num_rounds / cfg.simulation.eqivalent_cent_epochs
        ))
        print(f"Total dataset size: {total_steps}, num_unique_rounds: {num_unique_rounds}, eqivalent_cent_epochs: {cfg.simulation.eqivalent_cent_epochs}")
        train_data_splits = [train_data.shard(num_shards=num_unique_rounds, index=i) for i in range(num_unique_rounds)]
        return HuggingFaceClient(
            config=cfg,
            tokenizer=tokenizer,
            collator=collator,
            train_data=train_data_splits,
            eval_data=eval_set,
            lora_config=lora_config,
            client_id=partition_id,
        ).to_client()

    client_app = ClientApp(client_fn)

    # Server app initialisation
    def server_fn(context: Context):
        server_config = ServerConfig(num_rounds=cfg.simulation.num_rounds)
        strategy = FedAvg(
            min_available_clients=cfg.simulation.num_clients,
            on_fit_config_fn=fit_config_fn,
            evaluate_fn=get_evaluate_fn(
                cfg=cfg,
                tokenizer=tokenizer,
                collator=collator,
                eval_data=eval_set,
                lora_config=lora_config,
            ),
        )
        return ServerAppComponents(
            strategy=strategy,
            config=server_config,
            client_manager=client_manager,
        )

    server_app = ServerApp(server_fn=server_fn)

    # Run simulation
    run_simulation(
        client_app=client_app,
        server_app=server_app,
        num_supernodes=cfg.simulation.num_clients,
        backend_config={"client_resources": dict(cfg.simulation.client_resources)},
    )


if __name__ == "__main__":
    start_flower_simulation()
