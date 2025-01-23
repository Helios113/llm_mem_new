import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset
import tqdm

def plot_max_sequence_lengths(dataset_path: str, tokenizer_name: str, max_seq_length: int):
    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Tokenize dataset and calculate sequence lengths
    sequence_lengths = []
    for sample in tqdm.tqdm(dataset):
        tokens = tokenizer(sample['text'])
        sequence_lengths.append(len(tokens['input_ids']))

    # Plot bar plot of sequence lengths
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(sequence_lengths, edgecolor='black')
    plt.title(f'Distribution of Sequence Lengths in {os.path.dirname(dataset_path).split("/")[-1]}')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.grid(True)

    # Add text for bin ranges and counts
    for count, bin_start, bin_end in zip(counts, bins[:-1], bins[1:]):
        plt.text((bin_start + bin_end) / 2, count, f'{int(count)}\n[{int(bin_start)}, {int(bin_end)})', ha='center', va='bottom')

    # Save plot in the same directory as the dataset file
    plot_path = os.path.join(os.path.dirname(dataset_path), "sequence_lengths.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    dataset_path = "/nfs-share/pa511/new_work/data/amazonqa/data_train.json"
    tokenizer_name = "EleutherAI/pythia-125M"
    max_seq_length = 2048
    plot_max_sequence_lengths(dataset_path, tokenizer_name, max_seq_length)
