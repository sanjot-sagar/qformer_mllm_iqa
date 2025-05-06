import os
import json

base_path = '/scratch/sanjotst/datasets/LIVE_IQA/'
json_data = []

# Function to read MOS from info.txt files in each subdirectory


def read_mos_from_info_files(base_path):
    folders = ['fastfading', 'gblur', 'jp2k', 'jpeg', 'wn']
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            info_file = os.path.join(folder_path, 'info.txt')
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        image_name = parts[1]
                        mos = float(parts[2])
                        img_path = os.path.join(folder, image_name)
                        json_data.append({
                            "img_path": img_path,
                            # Placeholder for DMOS value (not used here)
                            "gt_score": None,
                            "mos_value": mos
                        })


# Call function to read MOS from info.txt files
read_mos_from_info_files(base_path)

# Path to save the JSON file
json_file_path = '/home/sanjotst/llm_iqa/llm-iqa/labels/live_iqa.json'

# Save JSON data to file
with open(json_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"JSON file saved successfully at: {json_file_path}")

# perfect just on problem
# this is how the json file looks like
#  {
    #     "img_path": "fastfading/img1.bmp",
    #     "gt_score": null,
    #     "mos_value": 16.5
    # },
    # {
    #     "img_path": "fastfading/img2.bmp",
    #     "gt_score": null,
    #     "mos_value": 18.9
    # },
    # {
    #     "img_path": "fastfading/img3.bmp",
    #     "gt_score": null,
    #     "mos_value": 21.3
    # },
    # {
    #     "img_path": "fastfading/img4.bmp",
    #     "gt_score": null,
    #     "mos_value": 23.7
    # },
    # copy each respective mos_value to gt_score
    # delete the mos_value parameter
    # write a new script for this, you already know the path of this json file