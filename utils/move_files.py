import os
import shutil

source_dir = "/scratch/sanjotst/live_fb_2_dist_5_level_full"
destination_dir = "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/annotator_matrix_testing"


# Get all subdirectories in alphabetical order
subdirs = sorted(next(os.walk(source_dir))[1])
# subdirs = sorted(os.walk(source_dir))
print(subdirs)
# Move the first 16 subdirectories
for subdir in subdirs[:16]:
    print("entering loop")
    source_path = os.path.join(source_dir, subdir)
    print(source_path)
    destination_path = os.path.join(destination_dir, subdir)
    shutil.move(source_path, destination_path)
    print(f"Moved directory: {source_path} to {destination_path}")
