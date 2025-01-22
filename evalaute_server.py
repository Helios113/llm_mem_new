from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from flwr.common.typing import NDArrays, Scalar
from flwr.common.logger import log

from typing import Dict, Tuple
import wandb
import torch
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from omegaconf import OmegaConf
import numpy as np
import random
import logging
from utils import gen_compute_metrics, GlobalStepCallback
import os

# Fix random seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def get_evaluate_fn(cfg, tokenizer, collator, eval_data, lora_config, client_steps):
    """Return an evaluation function for saving global model."""
    base_model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    model = get_peft_model(base_model, lora_config)
    def set_parameters(parameters):
        state_dict = model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        model.load_state_dict(state_dict)
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        with wandb.init(
            project=cfg.project,
            reinit=True,
            resume="allow",
            group=cfg.run_id,
            name=f"{cfg.run_id}-server",
            id=f"{cfg.run_id}-server",
            config=OmegaConf.to_object(cfg)
        ) as run:
            set_parameters(parameters)
            training_args = SFTConfig(
                output_dir=cfg.output_dir,
                run_name=f"{cfg.run_id}-server",
                **OmegaConf.to_object(cfg.training)
            )
            global_step_callback = GlobalStepCallback(
                elapsed_steps=server_round*client_steps, client_id=f"{cfg.run_id}-server", is_training=False
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                eval_dataset=eval_data,
                data_collator=collator,
                processing_class=tokenizer,
                compute_metrics=gen_compute_metrics(tokenizer),
                callbacks=[global_step_callback],
            )
            
            trainer.evaluate()
            global_step_callback.save_logs(cfg.output_dir)
            # Save checkpoint
            checkpoint_dir = os.path.join(cfg.output_dir, f"checkpoint-{server_round}")
            trainer.save_model(checkpoint_dir)
        # wandb.finish(0)
        return trainer.state.log_history[-1]["eval_loss"], {"perplexity":trainer.state.log_history[-1]["eval_perplexity"], "rouge1": trainer.state.log_history[-1]["eval_rouge1"], "f1": trainer.state.log_history[-1]["eval_f1"]}

    return evaluate

