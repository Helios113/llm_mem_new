# Huggingface imports
import csv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# Generic imports
import logging
import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import random

# Our packages
from data_formatting import get_tokenizer_and_data_collator_and_prompt_formatting
from utils import gen_compute_metrics, GlobalStepCallback

# Fix random seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

@hydra.main(version_base=None, config_path="config", config_name="config_cent")
def start_centralised_training(cfg: DictConfig):
    tokenizer, collator = get_tokenizer_and_data_collator_and_prompt_formatting(
        cfg.model.name, cfg.model.tokenizer, cfg.model.instruction_token
    )

    train_dataset = load_dataset(
        "json",
        data_files=cfg.train_dataset.files,
        split="train",
    )

    eval_dataset = load_dataset(
        "json",
        data_files=cfg.validation_dataset.files,
        split=cfg.validation_dataset.split,
    )

    with open_dict(cfg):
        cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Append run_id and output_dir to a table at a predefined path
    table_path = "/nfs-share/pa511/llm_memorisation/new_work/table.csv"
    with open(table_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cfg.run_id, cfg.output_dir])

    # Create LoraConfig
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
    )

    base_model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    model = get_peft_model(base_model, lora_config)

    with wandb.init(
        project=cfg.wandb.project,
        reinit=True,
        resume="allow",
        group=cfg.run_id,
        name=f"{cfg.run_id}-centralised",
        id=f"{cfg.run_id}-centralised",
        config=OmegaConf.to_object(cfg.wandb)
    ) as run:
        training_args = SFTConfig(
            output_dir=cfg.output_dir,
            run_name=f"{cfg.run_id}-centralised",
            **OmegaConf.to_object(cfg.training)
        )
        global_step_callback = GlobalStepCallback(
            current_round=0, steps_per_round=cfg.training.max_steps, client_id=f"{cfg.run_id}-centralised"
        )

        trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
                processing_class=tokenizer,
                compute_metrics=gen_compute_metrics(tokenizer),
                callbacks=[global_step_callback],
            )

        trainer.train()
        global_step_callback.save_logs(cfg.output_dir)
        # Save checkpoint
        checkpoint_dir = os.path.join(cfg.output_dir, f"checkpoint-fin")
        trainer.save_model(checkpoint_dir)

if __name__ == "__main__":
    start_centralised_training()
