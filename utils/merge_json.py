import json
import argparse

def merge_json_lists(file1: str, file2: str, output_file: str):
    dataset1 = json.load(open(file1))
    dataset2 = json.load(open(file2))
    merged_data = dataset1 + dataset2

    with open(output_file, "w") as f:
        json.dump(merged_data, f)
    
if __name__ == "__main__":
    
    file1 = "data/amazonqa/merge1.json"
    file2 = "data/amazonqa/test-qar_list_clean.json"
    output_file = "data/amazonqa/merge2.json"
    merge_json_lists(file1, file2, output_file)