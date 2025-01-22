import json
import argparse

def remove_empty_entries(json_file: str, key: str, output_file: str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    filtered_data = [entry for entry in data if entry.get(key) != ""]
    
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    
    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove entries with empty string values at a given key from a JSON list")
    parser.add_argument("json_file", type=str, help="Path to the input JSON file")
    parser.add_argument("key", type=str, help="Key to check for empty string values")
    parser.add_argument("output_file", type=str, help="Path to the output JSON file")
    args = parser.parse_args()
    
    remove_empty_entries(args.json_file, args.key, args.output_file)
