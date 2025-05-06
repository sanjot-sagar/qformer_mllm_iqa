import os
import json

# Paths to directories and MOS text files
nnid_base_dir = '/scratch/sanjotst/datasets/NNID'
sub512_dir = os.path.join(nnid_base_dir, 'sub512')
sub1024_dir = os.path.join(nnid_base_dir, 'sub1024')
sub2048_dir = os.path.join(nnid_base_dir, 'Sub2048')
mos512_file = os.path.join(nnid_base_dir, 'mos512_with_names.txt')
mos1024_file = os.path.join(nnid_base_dir, 'mos1024_with_names.txt')
mos2048_file = os.path.join(nnid_base_dir, 'mos2048_with_names.txt')

# Initialize list to store JSON objects
json_data = []

# Function to read MOS data from text file


def read_mos_data(mos_file):
    mos_data = {}
    with open(mos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                mos_value = float(parts[0])
                img_filename = parts[1].strip()
                mos_data[img_filename] = mos_value
    return mos_data


# Read MOS data from text files
mos512_data = read_mos_data(mos512_file)
mos1024_data = read_mos_data(mos1024_file)
mos2048_data = read_mos_data(mos2048_file)

# Function to create JSON data


def create_json_data(directory, mos_data):
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            if filename in mos_data:
                gt_score = mos_data[filename]
            else:
                gt_score = None  # Handle cases where MOS is not available
            json_data.append({
                # Remove nnid_base_dir from img_path
                'img_path': img_path.replace(nnid_base_dir + '/', ''),
                'gt_score': gt_score
            })
    return json_data


# Create JSON data for sub512 directory
json_data.extend(create_json_data(sub512_dir, mos512_data))

# Create JSON data for sub1024 directory
json_data.extend(create_json_data(sub1024_dir, mos1024_data))

# Create JSON data for sub2048 directory
json_data.extend(create_json_data(sub2048_dir, mos2048_data))

# Path to save the JSON file
json_file_path = '/scratch/sanjotst/datasets/NNID/nnid.json'

# Save JSON data to file
with open(json_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"JSON file created successfully at: {json_file_path}")
