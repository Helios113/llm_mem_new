import json

# Load the nested JSON file
with open('/nfs-share/pa511/new_work/data/pubmedqa/train-00000-of-00001.json', 'r') as file:
    data = json.load(file)

# Extract values into a list of dictionaries
extracted_data = []
for item in data:
    extracted_data.append({
        'question': item['question'],
        'answer': item['long_answer']
    })

# Save the extracted data to a new JSON file
with open('/nfs-share/pa511/new_work/data/pubmedqa/train.json', 'w') as file:
    json.dump(extracted_data, file, indent=4)
