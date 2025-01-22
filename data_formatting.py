from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
import numpy as np
import random
import torch

# Fix random seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def get_tokenizer_and_data_collator_and_prompt_formatting(model_name: str, tokenizer_name: str, instruction_token = False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
    response_template_with_context = '###Responce:'
    if instruction_token:
        instruction_template_with_context = '###Instruction:'
        data_collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template_with_context,response_template=response_template_with_context, tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template_with_context, tokenizer=tokenizer)
    return tokenizer, data_collator
