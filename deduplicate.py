import json
import argparse
from datasets import load_dataset, Dataset

def deduplicate_dataset(input_file: str, output_file: str, column: str):
    # Load the dataset
    dataset = load_dataset("json", data_files=input_file, split="train")
    total_records_before = len(dataset)

    # Deduplicate based on the given column
    unique_data = {entry[column]: entry for entry in dataset}.values()
    num_duplicates = total_records_before - len(unique_data)
    total_records_after = len(unique_data)

    # Convert to a Dataset object
    deduplicated_dataset = Dataset.from_dict({k: [dic[k] for dic in unique_data] for k in unique_data[0]})

    # Save the deduplicated dataset
    deduplicated_dataset.to_json(output_file)

    print(f"Total records before deduplication: {total_records_before}")
    print(f"Total records after deduplication: {total_records_after}")
    print(f"Number of duplicates found: {num_duplicates}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate JSON dataset based on a given column")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file")
    parser.add_argument("output_file", type=str, help="Path to save the deduplicated JSON file")
    parser.add_argument("column", type=str, help="Column name to deduplicate on")

    args = parser.parse_args()
    deduplicate_dataset(args.input_file, args.output_file, args.column)
