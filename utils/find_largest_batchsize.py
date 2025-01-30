import torch
import time
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import hydra
from omegaconf import DictConfig, open_dict
from data_formatting import get_tokenizer_and_data_collator_and_prompt_formatting
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from torch.utils.data import Dataset

class ArtificialDataset(Dataset):
    def __init__(self, tokenizer, max_seq_length, num_samples=1000):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        strn = "###Responce:"
        strn += "ONETOKEN" * self.max_seq_length
        tokens = self.tokenizer(strn, return_tensors="pt", padding="max_length", truncation=True)
        return {key: val.squeeze(0) for key, val in tokens.items()}

def sanitize_filename(name: str) -> str:
    return name.replace("/", "_")

def find_largest_batchsize_for_model(cfg: DictConfig, use_lora: bool, results: list):
    tokenizer, collator = get_tokenizer_and_data_collator_and_prompt_formatting(
        cfg.model.name, cfg.model.tokenizer, cfg.model.instruction_token
    )

    train_dataset = ArtificialDataset(tokenizer, max_seq_length=512)

    # Create LoraConfig if LoRA is enabled
    lora_config = None
    if use_lora:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    device_name = sanitize_filename(torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu")
    param_size = sum(p.numel() for p in model.parameters()) * 4 / (512 ** 3)  # Convert to GB (assuming float32)
    total_vram = torch.cuda.get_device_properties(device).total_memory / (512 ** 3) if torch.cuda.is_available() else "N/A"
    floating_point_accuracy = next(model.parameters()).dtype
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (512 ** 3)

    # Perform binary search to find the largest batch size that fits in memory
    low, high = 1, 512
    while low <= high:
        batch_size = (low + high) // 2
        try:
            training_args = SFTConfig(
                output_dir="./tmp",
                per_device_train_batch_size=batch_size,
                num_train_epochs=1,
                max_seq_length=512,
                max_steps=1,
                report_to="none"            
            )
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=collator,
                tokenizer=tokenizer,
            )
            start_time = time.time()
            trainer.train()
            end_time = time.time()
            throughput = end_time - start_time
            results.append({
                "batch_size": batch_size,
                "throughput": throughput,
                "device_name": device_name,
                "param_size_gb": param_size,
                "total_vram_gb": total_vram,
                "floating_point_accuracy": floating_point_accuracy,
                "trainable_params": trainable_params,
                "lora_config_alpha": lora_config.lora_alpha if use_lora else "None",
                "lora_config_r": lora_config.r if use_lora else "None",
            })
            print(f"Batch size {batch_size} fits in memory")
            low = batch_size + 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                high = batch_size - 1
            else:
                raise e

    return high

@hydra.main(version_base=None, config_path="config", config_name="config_cent")
def find_largest_batchsize(cfg: DictConfig):
    results = []
    batch_size_with_lora = find_largest_batchsize_for_model(cfg, use_lora=True, results=results)
    batch_size_without_lora = find_largest_batchsize_for_model(cfg, use_lora=False, results=results)

    device_name = sanitize_filename(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
    os.makedirs("/nfs-share/pa511/new_work/results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(f"/nfs-share/pa511/new_work/results/{device_name}_{sanitize_filename(cfg.model.name)}_results.csv", index=False)

    print(f"Largest batch size with LoRA that fits in memory: {batch_size_with_lora}")
    print(f"Largest batch size without LoRA that fits in memory: {batch_size_without_lora}")

if __name__ == "__main__":
    find_largest_batchsize()
