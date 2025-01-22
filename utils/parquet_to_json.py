import pandas as pd

# Specify the input Parquet file and the output JSON file
parquet_file = "data/pubmedqa/train-00000-of-00001.parquet"  # Replace with your Parquet file path
json_file = "data/pubmedqa/train-00000-of-00001.json"       # Replace with the desired JSON file path

# Load the Parquet file
df = pd.read_parquet(parquet_file)

# Save the DataFrame as a JSON file
df.to_json(json_file, orient="records", lines=True)  # Use lines=True for JSONL format
print(f"Parquet file has been converted to JSON and saved as {json_file}")
