from finetune import FederatedDataset
from transformers import AutoTokenizer
from datasets import load_dataset


def load_and_tokenize_datasets(config):
    # Load datasets
    
    train_dataset = load_dataset(
        "json",
        data_files=config['train_dataset']['files'],
        split="train",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer'])

    # Tokenize datasets
    train_tokens = [tokenizer.encode(text) for text in train_dataset[config['train_dataset']['text_field']]]

    # Calculate number of tokens
    num_train_tokens = sum(len(tokens) for tokens in train_tokens)
    num_train_entries = len(train_dataset)
    return num_train_tokens, num_train_entries


def calculate_steps_per_round(config, num_train_tokens):
    batchsize = config['training']['per_device_train_batch_size']
    number_of_clients = config['simulation']['num_clients']
    client_participation_ratio = config['simulation'].get('client_participation_ratio', 1.0)
    rounds = config['simulation']['num_rounds']
    
    # Calculate the number of steps per round needed
    total_batches_per_round = (number_of_clients * client_participation_ratio) * batchsize
    steps_per_round = num_train_tokens / (rounds*total_batches_per_round)

    return steps_per_round

if __name__ == "__main__":
    import yaml
    with open('/nfs-share/pa511/llm_memorisation/new_work/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    num_train_tokens, num_train_entries = load_and_tokenize_datasets(config)
    print(f"Number of tokens in training dataset: {num_train_tokens}")
    print(f"Number of training entries: {num_train_entries}")

    steps_per_round = calculate_steps_per_round(config, num_train_tokens)
    print(f"Steps per round needed to see all training tokens: {steps_per_round}")
