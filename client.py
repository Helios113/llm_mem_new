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

def custom_lr_scheduler(optimizer, num_training_steps, num_warmup_steps, current_round, total_rounds):
    def lr_lambda(current_step):
        global_step = (current_round-1) * num_training_steps + current_step
        if global_step < num_warmup_steps:
            return float(global_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps * total_rounds - global_step) / float(max(1, num_training_steps * total_rounds - num_warmup_steps))
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
        self.eval_data = eval_data

        # Initialize Weights & Biases
        self.client_id = client_id

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
            config=OmegaConf.to_object(self.cfg.wandb)
        ) as run:
            self.set_parameters(parameters)
            # ...existing code...

            current_round = int(config["current_round"])
            training_args = SFTConfig(
                output_dir=self.cfg.output_dir,
                run_name=f"{self.cfg.run_id}-client-{self.client_id}",
                **OmegaConf.to_object(self.cfg.training)
            )
            log(logging.INFO, "current round: {}".format(current_round))
            global_step_callback = GlobalStepCallback(
                current_round=current_round, steps_per_round=self.cfg.training.max_steps, client_id = f"{self.cfg.run_id}-client-{self.client_id}"
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_data[current_round - 1],
                eval_dataset=self.eval_data,
                data_collator=self.collator,
                processing_class=self.tokenizer,
                compute_metrics=gen_compute_metrics(self.tokenizer),
                callbacks=[global_step_callback],
            )

            num_training_steps = self.cfg.training.max_steps
            num_warmup_steps = int(self.cfg.training.warmup_ratio * num_training_steps * self.cfg.simulation.num_rounds)
            optimizer = trainer.create_optimizer()
            lr_scheduler = custom_lr_scheduler(optimizer, num_training_steps, num_warmup_steps, current_round, self.cfg.simulation.num_rounds)
            trainer.lr_scheduler = lr_scheduler

            trainer.train()
            global_step_callback.save_logs(self.cfg.output_dir)
        return self.get_parameters(config), len(self.train_data[current_round - 1]), {}
