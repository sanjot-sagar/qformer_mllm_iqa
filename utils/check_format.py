import csv
import os

# Path to the CSV file
csv_file = "/home/sanjotst/llm_iqa/llm-iqa/labels/live_fb_split.csv"

# Function to extract the format from the image name


def extract_format(image_name):
    return os.path.splitext(image_name)[1]


# Set to store unique formats
unique_formats = set()

# Read the CSV file and extract formats
with open(csv_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['name_image']
        image_format = extract_format(image_name)
        # Convert to lowercase for consistency
        unique_formats.add(image_format.lower())

# Print the unique formats
for fmt in unique_formats:
    print(fmt)
