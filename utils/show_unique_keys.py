import json
import argparse

def show_unique_keys(json_file: str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    unique_keys = set()
    for entry in data:
        unique_keys.update(entry.keys())
    
    print("Unique keys in the JSON file:")
    for key in unique_keys:
        print(key)

if __name__ == "__main__":
    json_file = "/nfs-share/pa511/new_work/data/pubmedqa/train-00000-of-00001.json"
    
    show_unique_keys(json_file)
