import datetime
import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import csv
import os
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from utils import gen_compute_metrics, GlobalStepCallback
from data_formatting import get_tokenizer_and_data_collator_and_prompt_formatting
from trl import SFTTrainer, SFTConfig
import copy


# Fix random seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)


def constant_with_cooloff_lr_scheduler(
    optimizer, num_warmup_steps, elapsed_steps, total_steps, cooloff_steps
):
    def lr_lambda(current_step):
        global_step = elapsed_steps + current_step
        if global_step < num_warmup_steps:
            return float(global_step) / float(max(1, num_warmup_steps))
        elif global_step < total_steps - cooloff_steps:
            return 1.0
        return max(
            0.0,
            float(total_steps - global_step) / float(max(1, cooloff_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def generate_run_id(cfg: DictConfig, unique_id=None):
    model_name = cfg.model.name.split("/")[-1]
    dataset_name = cfg.dataset.path.split("/")[-1]
    lora_status = "lora" if cfg.simulation.use_lora else "no_lora"
    if unique_id is None:
        unique_id = wandb.util.generate_id()
    return f"{model_name}_{dataset_name}_{lora_status}_{unique_id}_flsim"


def generate_run_name(client_id, run_id, dataset_name, lora_status):
    return f"client_{client_id}_{run_id}_{dataset_name}_{lora_status}"


def aggregate_parameters(client_parameters):
    avg_params = client_parameters[0]
    for key in avg_params.keys():
        for i in range(1, len(client_parameters)):
            avg_params[key] += client_parameters[i][key]
        avg_params[key] = avg_params[key] / len(client_parameters)
    return avg_params


def evaluate_model(
    model, state_dict, eval_set, collator, tokenizer, cfg, elapsed_steps, now
):
    with wandb.init(
        project=cfg.project,
        reinit=True,
        resume="allow",
        group=cfg.run_id,
        name=f"{cfg.run_id}_server",
        id=f"{cfg.run_id}_server",
        config=OmegaConf.to_object(cfg),
    ) as run:
        model.load_state_dict(state_dict)
        training_args = SFTConfig(
            output_dir=cfg.output_dir,
            per_device_eval_batch_size=cfg.training.per_device_train_batch_size,
            **OmegaConf.to_object(cfg.training),
        )
        global_step_callback = GlobalStepCallback(
            elapsed_steps=elapsed_steps,
            client_id=f"{cfg.run_id}_server",
            is_training=False,
            start_time=now,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            eval_dataset=eval_set,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=gen_compute_metrics(tokenizer),
            callbacks=[global_step_callback],
        )
        
        eval_results = trainer.evaluate()
        global_step_callback.save_logs(cfg.output_dir)
        return eval_results


@hydra.main(version_base=None, config_path="config", config_name="config")
def simulate_federated_learning(cfg: DictConfig):
    now = datetime.datetime.now()

    tokenizer, collator = get_tokenizer_and_data_collator_and_prompt_formatting(
        cfg.model.name, cfg.model.tokenizer, cfg.model.instruction_token
    )

    # Load the train dataset
    train_set = load_dataset(
        "json",
        data_files=os.path.join(cfg.dataset.path, "data_train.json"),
        split="train",
    )

    # Load the eval dataset
    eval_set = load_dataset(
        "json",
        data_files=os.path.join(cfg.dataset.path, "data_non_member.json"),
        split="train",
    )
    if len(eval_set) > 1000:
        eval_set = eval_set.select(range(1000))

    # Split train_data into equal sections
    num_rounds = cfg.simulation.num_rounds
    total_steps = len(train_set)
    num_unique_rounds = int(np.ceil(num_rounds / cfg.simulation.eqivalent_cent_epochs))
    client_remainder = total_steps % cfg.simulation.num_clients
    client_data_size = total_steps // cfg.simulation.num_clients
    train_set = train_set.select(range(total_steps - client_remainder))

    client_round_data_size = client_data_size // num_unique_rounds
    cleint_ds_remainder = client_round_data_size % num_unique_rounds
    total_loss = cleint_ds_remainder * cfg.simulation.num_clients + client_remainder
    client_round_data_size = int(
        np.ceil(client_round_data_size / cfg.training.per_device_train_batch_size)
    )
    total_steps = client_round_data_size * num_rounds

    print(
        f"Total centralised equivalent data size: {client_round_data_size * num_rounds * cfg.simulation.num_clients}"
    )
    print(f"Total lossed elements: {total_loss}")
    print(f"Total expected steps per round: {client_round_data_size}")
    print(f"Total unique rounds: {num_unique_rounds}")
    print(f"Total steps: {total_steps}")
    print(f"Total client data size: {client_data_size}")
    print(f"Learning rate: {cfg.training.learning_rate}")

    if "run_id" not in cfg:
        if cfg.resume:
            raise ValueError("Run ID must be provided when resuming training")
        with open_dict(cfg):
            cfg.run_id = generate_run_id(cfg, None)
    else:
        with open_dict(cfg):
            if cfg.run_id == "":
                cfg.run_id = generate_run_id(cfg, None)
            else:
                cfg.run_id = generate_run_id(cfg, cfg.run_id)

    evals_per_round = int(cfg.num_loggings / num_rounds) - 1
    evals_per_round = 1 if evals_per_round < 1 else evals_per_round
    logging_steps = int(client_round_data_size / evals_per_round)

    with open_dict(cfg):
        cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        cfg.evaluation.per_device_eval_batch_size = (
            cfg.training.per_device_train_batch_size
        )
        cfg.training.logging_steps = logging_steps
        cfg.training.eval_steps = cfg.training.logging_steps

    start_round = 0
    if cfg.resume:
        start_round = cfg.start_round

    table_path = "/nfs-share/pa511/new_work/table.csv"
    with open(table_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([cfg.run_id, cfg.output_dir])

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

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    if lora_config:
        model = get_peft_model(model, lora_config)

    optimizer = SGD(model.parameters(), lr=cfg.training.learning_rate)
    num_warmup_steps = int(cfg.simulation.warmup_ratio * total_steps)
    cooloff_steps = int(cfg.simulation.cooloff_ratio * total_steps)

    # Initialize the model parameters for the first round
    round_parameters = model.state_dict()
    elapsed_steps = 0
    for round_num in range(start_round, num_rounds):
        print(f"Starting round {round_num + 1}/{num_rounds}")

        elapsed_steps = round_num * client_round_data_size
        print(f"Elapsed steps: {elapsed_steps}")
        client_parameters = []
        for client_id in range(cfg.simulation.num_clients):
            # Initialize wandb for each client
            with wandb.init(
                project=cfg.project,
                reinit=True,
                resume="allow",
                group=cfg.run_id,
                name=f"{cfg.run_id}-client-{client_id}",
                id=f"{cfg.run_id}-client-{client_id}",
                config=OmegaConf.to_object(cfg),
            ) as run:
                # Load the round parameters
                model.load_state_dict(round_parameters)
                client_data = train_set.shard(
                    num_shards=cfg.simulation.num_clients, index=client_id
                )
                client_data = client_data.shard(
                    num_shards=num_unique_rounds, index=round_num % num_unique_rounds
                )
                training_args = SFTConfig(
                    output_dir=cfg.output_dir,
                    per_device_eval_batch_size=cfg.training.per_device_train_batch_size,
                    **OmegaConf.to_object(cfg.training),
                )

                global_step_callback = GlobalStepCallback(
                    elapsed_steps=elapsed_steps,
                    client_id=f"{cfg.run_id}-client-{client_id}",
                    start_time=now,
                )
                lr_scheduler = constant_with_cooloff_lr_scheduler(
                    optimizer,
                    num_warmup_steps,
                    elapsed_steps,
                    total_steps,
                    cooloff_steps,
                )
                trainer = SFTTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=client_data,
                    eval_dataset=eval_set,
                    data_collator=collator,
                    tokenizer=tokenizer,
                    compute_metrics=gen_compute_metrics(tokenizer),
                    callbacks=[global_step_callback],
                    optimizers=(optimizer, lr_scheduler),
                )
                trainer.train()
                global_step_callback.save_logs(cfg.output_dir)

                # Save client parameters
                client_parameters.append(copy.deepcopy(model.state_dict()))

        # Aggregate the model parameters
        round_parameters = aggregate_parameters(client_parameters)

        # Evaluate the model at the end of the round
        eval_results = evaluate_model(
            model,
            round_parameters,
            eval_set,
            collator,
            tokenizer,
            cfg,
            (round_num+1) * client_round_data_size,
            now,
        )
        print(f"Round {round_num + 1} evaluation results: {eval_results}")


if __name__ == "__main__":
    simulate_federated_learning()
