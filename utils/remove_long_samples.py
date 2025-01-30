import os
import json
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import tqdm

def remove_long_samples(dataset_path: str, tokenizer_name: str, max_seq_length: int, threshold: int, output_path: str):
    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Filter out samples with sequence lengths above the threshold
    filtered_samples = []
    for sample in tqdm.tqdm(dataset):
        tokens = tokenizer(sample['text'], truncation=True, max_length=max_seq_length)
        if len(tokens['input_ids']) <= threshold:
            filtered_samples.append(sample)

    # Save the filtered dataset as a JSON file
    filtered_dataset = Dataset.from_dict({"text": [sample['text'] for sample in filtered_samples]})
    filtered_dataset.to_json(output_path)
    print(f"Filtered dataset saved to {output_path}")

if __name__ == "__main__":
    dataset_path = "/nfs-share/pa511/new_work/data/triviaqa/raw/data_train.json"
    tokenizer_name = "EleutherAI/pythia-125M"
    max_seq_length = 2048
    threshold = 512
    output_path = "/nfs-share/pa511/new_work/data/triviaqa/raw/data_train_filtered.json"
    remove_long_samples(dataset_path, tokenizer_name, max_seq_length, threshold, output_path)
