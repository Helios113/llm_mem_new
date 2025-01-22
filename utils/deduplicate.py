import json
import argparse
import os
from datasets import load_dataset, Dataset

def find_longest_sequence_length(dataset, column):
    return max(len(entry[column]) for entry in dataset)

def deduplicate_dataset(input_file: str, output_file: str, column: str):
    # Load the dataset
    dataset = load_dataset("json", data_files=input_file, split="train")
    total_records_before = len(dataset)

    # Find the longest sequence length

    # Deduplicate based on the given column
    unique_data = {entry[column]: entry for entry in dataset}
    num_duplicates = total_records_before - len(unique_data)
    total_records_after = len(unique_data)
    
    new_data = list(unique_data.values())
    print(new_data[0])
    # Convert to a Dataset object
    deduplicated_dataset = Dataset.from_list(new_data)

    # Save the deduplicated dataset
    deduplicated_dataset.to_json(output_file)
    longest_sequence_length = find_longest_sequence_length(deduplicated_dataset, column)
    
    # Save dataset info to data.txt
    info_file = os.path.join(os.path.dirname(output_file), "data.txt")
    with open(info_file, "w") as f:
        f.write(f"Total records before deduplication: {total_records_before}\n")
        f.write(f"Total records after deduplication: {total_records_after}\n")
        f.write(f"Number of duplicates found: {num_duplicates}\n")
        f.write(f"Longest sequence length in column '{column}': {longest_sequence_length}\n")

    print(f"Total records before deduplication: {total_records_before}")
    print(f"Total records after deduplication: {total_records_after}")
    print(f"Number of duplicates found: {num_duplicates}")
    print(f"Longest sequence length in column '{column}': {longest_sequence_length}")

if __name__ == "__main__":
    input_file = "/nfs-share/pa511/new_work/data/pubmedqa/train.json"
    output_file ="/nfs-share/pa511/new_work/data/pubmedqa/train_de.json"
    column = "question"
    deduplicate_dataset(input_file, output_file, column)
