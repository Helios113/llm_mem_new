import json
import random
import os
from typing import List, Tuple
from datasets import load_dataset

def formatting_prompts_func(example, key1, key2):
        output_texts = []
        # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
        for i in range(len(example)):
            text = "~ins~ "+example[i][key1]+"~res~ "+example[i][key2]
            # o = {key1: example[i][key1], key2: example[i][key2], 'text': text}
            o = {'text': text}
            output_texts.append(o)
        return output_texts
    


def generate_datasets_with_merge(
    dataset_path: str,
    save_dir: str,
    x: float,
    col1: str,
    col2: str
) -> Tuple[List[dict], List[dict], List[dict]]:

    target_name = "data"
    # Load the dataset
    # with open(dataset_path, 'r') as file:
    #     dataset = json.load(file)
    dataset = load_dataset("json", data_files=dataset_path, split="train").to_list()
    
    # Shuffle the dataset to ensure randomness
    random.shuffle(dataset)
    
    # Calculate sample size
    sample_size = int(x * len(dataset))
    
    # Split into non-member set and rest of the dataset (train candidate set)
    # format_dataset = formatting_prompts_func(dataset, col1, col2)
    format_dataset = dataset
    non_member_set = format_dataset[:sample_size]
    train_candidate_set = format_dataset[sample_size:]
    
    # Shuffle train candidate set and select member set from it
    random.shuffle(train_candidate_set)
    member_set = train_candidate_set[:sample_size]
    
    # Remaining becomes the final train set
    train_set = train_candidate_set
    
    # # Function to merge columns and discard the rest
    
    
    # # # Process non-member and member sets
    # non_member_set = formatting_prompts_func(non_member_set, col1, col2)
    # member_set = formatting_prompts_func(member_set, col1, col2)
    
    # Automatically generate output file paths
    non_member_file = os.path.join(save_dir, f"{target_name}_non_member.json")
    train_file = os.path.join(save_dir, f"{target_name}_train.json")
    member_file = os.path.join(save_dir, f"{target_name}_member.json")
    
    # Save sets to JSON files
    with open(non_member_file, 'w') as file:
        json.dump(non_member_set, file, indent=4)
    
    with open(train_file, 'w') as file:
        json.dump(train_set, file, indent=4)
    
    with open(member_file, 'w') as file:
        json.dump(member_set, file, indent=4)
    
    print(f"Generated datasets:")
    print(f"Non-member set saved to: {non_member_file}")
    print(f"Train set saved to: {train_file}")
    print(f"Member set saved to: {member_file}")
    
    return non_member_set, train_set, member_set

# Usage example:
target_name = "/nfs-share/pa511/new_work/data/amazonqa/raw/data_train_filtered.json"
save_dir = "data/amazonqa"
# target_name = "/nfs-share/pa511/new_work/data/aus_qa/raw/train_de.json"
# save_dir = "data/aus_qa/raw"
col1 = "question"
col2 = "answer"

x = 0.1 # 10% sampling
generate_datasets_with_merge(target_name, save_dir,x, col1, col2)
# /nfs-share/pa511/llm_memorisation/datasets_our/medical_dataset/deduplicated_medical_meadow_flashcards/deduplicated_medical_meadow_flashcards.json