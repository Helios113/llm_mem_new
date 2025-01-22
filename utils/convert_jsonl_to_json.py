import json
import argparse

def convert_jsonl_to_json(jsonl_file: str, json_file: str):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Converted {jsonl_file} to {json_file}")

if __name__ == "__main__":
    input_file = "data/triviaqa/train_de.json"
    output_file = "data/triviaqa/train_de.json"
    convert_jsonl_to_json(input_file, output_file)
