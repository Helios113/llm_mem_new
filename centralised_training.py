# Huggingface imports
import csv
import datetime
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import SGD
# Generic imports
import logging
import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import random
import yaml

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

def constant_with_cooloff_lr_scheduler(optimizer, num_warmup_steps, total_steps, cooloff_steps):
    def lr_lambda(current_step):
        global_step = current_step
        if global_step < num_warmup_steps:
            return float(global_step) / float(max(1, num_warmup_steps))
        elif global_step < total_steps - cooloff_steps:
            return 1.0
        return max(
            0.0,
            float(total_steps - global_step) / float(max(1, cooloff_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)

def generate_run_id(cfg: DictConfig):
    model_name = cfg.model.name.split("/")[-1]
    dataset_name = cfg.dataset.path.split("/")[-1]
    lora_status = "lora" if cfg.simulation.use_lora else "no_lora"
    unique_id = wandb.util.generate_id()
    return f"{model_name}_{dataset_name}_{lora_status}_{unique_id}"

@hydra.main(version_base=None, config_path="config", config_name="config_cent")
def start_centralised_training(cfg: DictConfig):
    now = datetime.datetime.now()
    tokenizer, collator = get_tokenizer_and_data_collator_and_prompt_formatting(
        cfg.model.name, cfg.model.tokenizer, cfg.model.instruction_token
    )
    

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
    if len(eval_set) >1000:
        eval_set = eval_set.select(range(1000))
    
    datacard_path = os.path.join(cfg.dataset.path, "datacard.yaml")
    with open(datacard_path, 'r') as file:
        datacard = yaml.safe_load(file)
    centralised_steps = datacard.get("centralised_steps", 0)
    with open_dict(cfg):
        cfg.training.max_steps = int(np.ceil(centralised_steps/cfg.training.per_device_train_batch_size))
        cfg.training.logging_steps = int(cfg.training.max_steps/100)
        

    # Append run_id and output_dir to a table at a predefined path
    table_path = "/nfs-share/pa511/llm_memorisation/new_work/table.csv"
    
    if "run_id" not in cfg:
        if cfg.resume:
            raise ValueError("Run ID must be provided when resuming training")
        with open_dict(cfg):
            cfg.run_id = generate_run_id(cfg)
    with open_dict(cfg):
        cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(table_path, mode='a', newline='') as file:
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

    base_model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    if lora_config:
        model = get_peft_model(base_model, lora_config)
    else:
        model = base_model

    total_steps = cfg.training.max_steps
    num_warmup_steps = int(total_steps * cfg.simulation.warmup_ratio)
    cooloff_steps = int(total_steps * cfg.simulation.cooloff_ratio)
    with wandb.init(
        project=cfg.project,
        reinit=True,
        resume="allow",
        group=cfg.run_id,
        name=f"{cfg.run_id}-centralised",
        id=f"{cfg.run_id}-centralised",
        config=OmegaConf.to_object(cfg)
    ) as run:
        training_args = SFTConfig(
            output_dir=cfg.output_dir,
            run_name=f"{cfg.run_id}-centralised",
            **OmegaConf.to_object(cfg.training),
        )
        global_step_callback = GlobalStepCallback(
            elapsed_steps=0, client_id=f"{cfg.run_id}-centralised", start_time=now
        )

        optimizer = SGD(model.parameters(), lr=cfg.training.learning_rate)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=eval_set,
            data_collator=collator,
            processing_class=tokenizer,
            compute_metrics=gen_compute_metrics(tokenizer),
            callbacks=[global_step_callback],
            optimizers=(optimizer, constant_with_cooloff_lr_scheduler(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                total_steps=total_steps,
                cooloff_steps=cooloff_steps,
            )),
        )
        if cfg.resume:
            if cfg.checkpoint_path == "":
                raise ValueError("Checkpoint path must be provided when resuming training")
            trainer.train(resume_from_checkpoint=cfg.checkpoint_path)
        else:
            trainer.train()
        global_step_callback.save_logs(cfg.output_dir)

if __name__ == "__main__":
    start_centralised_training()
