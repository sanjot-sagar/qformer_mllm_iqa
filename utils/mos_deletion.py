import os
import json

# Path to the existing JSON file
json_file_path = '/home/sanjotst/llm_iqa/llm-iqa/labels/live_iqa.json'

# Load JSON data from file
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

# Iterate through JSON data and modify as required
for item in json_data:
    # Copy mos_value to gt_score and remove mos_value
    item['gt_score'] = item.pop('mos_value')

# Save modified JSON data back to file
with open(json_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"JSON file updated successfully at: {json_file_path}")
