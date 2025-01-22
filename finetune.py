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
from logging import DEBUG
import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, open_dict
import csv
import random
import os

# Our packages
from client import HuggingFaceClient
from data_formatting import get_tokenizer_and_data_collator_and_prompt_formatting
from client_manager import RandomOrgClientManager
from evalaute_server import get_evaluate_fn
from startegy import SaveModelStrategy


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


def generate_run_id(cfg: DictConfig):
    model_name = cfg.model.name.split("/")[-1]
    dataset_name = cfg.dataset.path.split("/")[-1]
    lora_status = "lora" if cfg.simulation.use_lora else "no_lora"
    unique_id = wandb.util.generate_id()
    return f"{model_name}_{dataset_name}_{lora_status}_{unique_id}"


@hydra.main(version_base=None, config_path="config", config_name="config")
def start_flower_simulation(cfg: DictConfig):

    client_manager = RandomOrgClientManager("2e4c64c1-80ab-42b9-8924-9576140f571e")
    tokenizer, collator = get_tokenizer_and_data_collator_and_prompt_formatting(
        cfg.model.name, cfg.model.tokenizer, cfg.model.instruction_token
    )

    # Load the train dataset
    train_set = load_dataset(
        "json",
        data_files=os.path.join(cfg.dataset.path, "data_train.json"),
        split="train"
    )

    # Load the eval dataset
    eval_set = load_dataset(
        "json",
        data_files=os.path.join(cfg.dataset.path, "data_non_member.json"),
        split="train"
    )
   
    # Split train_data into equal sections
    num_rounds = cfg.simulation.num_rounds
    total_steps = len(train_set)
    num_unique_rounds = int(np.ceil(num_rounds / cfg.simulation.eqivalent_cent_epochs))
    client_remainder = total_steps % cfg.simulation.num_clients
    client_data_size = total_steps // cfg.simulation.num_clients
    # Remove client_remainder elements from train set while keeping the type as a dataset
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

    if "run_id" not in cfg:
        with open_dict(cfg):
            cfg.run_id = generate_run_id(cfg)
    with open_dict(cfg):
        cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Append run_id and output_dir to a table at a predefined path
    table_path = "/nfs-share/pa511/llm_memorisation/new_work/table.csv"
    with open(table_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([cfg.run_id, cfg.output_dir])

    # Create LoraConfig if LoRA is enabled
    lora_config = None
    if cfg.simulation.use_lora:
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
        nonlocal num_unique_rounds
        nonlocal train_set
        partition_id = int(context.node_config["partition-id"])
        train_data = train_set.shard(
            num_shards=cfg.simulation.num_clients, index=partition_id
        )

        total_steps = len(train_data)
        remainder = total_steps % num_unique_rounds
        train_data = train_data.select(range(total_steps - remainder))

        train_data_splits = [
            train_data.shard(num_shards=num_unique_rounds, index=i)
            for i in range(num_unique_rounds)
        ]
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
        strategy = SaveModelStrategy(
            min_available_clients=cfg.simulation.num_clients,
            on_fit_config_fn=fit_config_fn,
            on_evaluate_config_fn=None,
            score_key = "rouge1",
            cfg = cfg,
            evaluate_fn=get_evaluate_fn(
                cfg=cfg,
                tokenizer=tokenizer,
                collator=collator,
                eval_data=eval_set,
                lora_config=lora_config,
                client_steps=client_round_data_size,
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
