import csv
import json

# Path to the CSV file and where to save the JSON file
csv_file = '/scratch/sanjotst/datasets/LIVE_IQA_new/LIVE_IQA_full.csv'
json_file_path = '/scratch/sanjotst/datasets/LIVE_IQA_new/live_iqa_new.json'

# Initialize list to store JSON objects
json_data = []

# Read data from CSV file and convert to JSON format
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_path = row['im_loc']
        gt_score = float(row['mos'])
        json_data.append({
            'img_path': img_path,
            'gt_score': gt_score
        })

# Save JSON data to file
with open(json_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"JSON file created successfully at: {json_file_path}")
