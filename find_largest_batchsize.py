import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import hydra
from omegaconf import DictConfig, open_dict
from data_formatting import get_tokenizer_and_data_collator_and_prompt_formatting

@hydra.main(version_base=None, config_path="config", config_name="config_cent")
def find_largest_batchsize(cfg: DictConfig):
    tokenizer, collator = get_tokenizer_and_data_collator_and_prompt_formatting(
        cfg.model.name, cfg.model.tokenizer, cfg.model.instruction_token
    )

    train_dataset = load_dataset(
        "json",
        data_files=cfg.train_dataset.files,
        split="train",
    )

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
        # Freeze base model weights
        for param in base_model.parameters():
            param.requires_grad = False
    else:
        model = base_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Find the largest batch size that fits in memory
    batch_size = 2
    while True:
        try:
            inputs = tokenizer(train_dataset["text"][:batch_size], return_tensors="pt", padding=True, truncation=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                model(**inputs)
            batch_size += 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size -= 2
                break
            else:
                raise e

    print(f"Largest batch size that fits in memory: {batch_size}")

if __name__ == "__main__":
    find_largest_batchsize()
