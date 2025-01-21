# Flower client imports
from flwr.client import NumPyClient
from flwr.client import ClientApp

# Flower other imports
from flwr.simulation import run_simulation
from flwr.common import Context
from flwr.common.logger import log

# Huggingface imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from utils import gen_compute_metrics, GlobalStepCallback
from omegaconf import OmegaConf

# Generic imports
import logging
import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig
import copy
from torch.optim.lr_scheduler import LambdaLR


# Load dataset and tokenize
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)


# def cosine_annealing(
#     current_round: int,
#     total_round: int,
#     lrate_max: float = 0.001,
#     lrate_min: float = 0.0,
# ) -> float:
#     """Implement cosine annealing learning rate schedule."""

#     cos_inner = math.pi * current_round / total_round
#     return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def custom_lr_scheduler(optimizer, num_warmup_steps, elapsed_steps, total_steps):
    def lr_lambda(current_step):
        global_step = elapsed_steps + current_step
        if global_step < num_warmup_steps:
            return float(global_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(total_steps - global_step)
            / float(max(1, total_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


# Define the Hugging Face model and tokenizer
class HuggingFaceClient(NumPyClient):
    def __init__(
        self, config, train_data, eval_data, tokenizer, collator, lora_config, client_id
    ):
        self.tokenizer = tokenizer
        self.collator = collator
        self.base_model = AutoModelForCausalLM.from_pretrained(config.model.name)
        self.model = get_peft_model(self.base_model, lora_config)
        self.cfg = config
        self.train_data = train_data
        self.steps_per_round = len(train_data[0])
        self.bs = config.training.per_device_train_batch_size
        self.total_steps = 0
        for i in range(len(train_data)):
            self.total_steps += len(self.train_data[i])
        self.total_steps =int(np.ceil(self.total_steps* config.simulation.eqivalent_cent_epochs/self.bs))

        self.eval_data = eval_data
        # Initialize Weights & Biases
        self.client_id = client_id
        self.num_unique_rounds = int(
            np.ceil(
                config.simulation.num_rounds / config.simulation.eqivalent_cent_epochs
            )
        )
        print(f"Client {client_id} initialized")
        print(
            f"Total client steps (ds size * epochs): {self.total_steps}, num_unique_rounds: {self.num_unique_rounds}, eqivalent_cent_epochs: {config.simulation.eqivalent_cent_epochs}"
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        log(logging.INFO, "Client {}: Starting training".format(self.client_id))
        with wandb.init(
            project=self.cfg.wandb.project,
            reinit=True,
            resume="allow",
            group=self.cfg.run_id,
            name=f"{self.cfg.run_id}-client-{self.client_id}",
            id=f"{self.cfg.run_id}-client-{self.client_id}",
            config=OmegaConf.to_object(self.cfg.wandb),
        ) as run:
            self.set_parameters(parameters)

            current_round = int(config["current_round"])
            training_args = SFTConfig(
                output_dir=self.cfg.output_dir,
                run_name=f"{self.cfg.run_id}-client-{self.client_id}",
                **OmegaConf.to_object(self.cfg.training),
            )
            log(logging.INFO, "current round: {}".format(current_round))

            elapsed_steps = 0
            for i in range(current_round - 1):
                elapsed_steps += len(
                    self.train_data[(current_round - 1) % self.num_unique_rounds]
                )
            elapsed_steps = int(np.ceil( elapsed_steps/self.bs))
            print(f"Client {self.client_id} elapsed steps: {elapsed_steps}")
            print(f"Client {self.client_id} fit, current_round: {current_round}")
            print(f"Client {self.client_id} fit, current_shard: {(current_round - 1) % self.num_unique_rounds}, shard_size: {len(self.train_data[(current_round - 1) % self.num_unique_rounds])/self.bs}")
            
            global_step_callback = GlobalStepCallback(
                elapsed_steps=elapsed_steps,
                client_id=f"{self.cfg.run_id}-client-{self.client_id}",
            )
            
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_data[
                    (current_round - 1) % self.num_unique_rounds
                ],
                eval_dataset=self.eval_data,
                data_collator=self.collator,
                processing_class=self.tokenizer,
                compute_metrics=gen_compute_metrics(self.tokenizer),
                callbacks=[global_step_callback],
            )

            num_warmup_steps = int(self.cfg.training.warmup_ratio * self.total_steps)
            optimizer = trainer.create_optimizer()
            lr_scheduler = custom_lr_scheduler(
                optimizer,
                num_warmup_steps,
                elapsed_steps,
                self.total_steps,
            )
            trainer.lr_scheduler = lr_scheduler

            trainer.train()
            global_step_callback.save_logs(self.cfg.output_dir)
        return self.get_parameters(config), len(self.train_data[(current_round - 1) % self.num_unique_rounds]), {}
