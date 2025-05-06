import numpy as np

# File paths
file1_path = "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/annotated_fsim_matrices_complete_livefb/AVA__1046.bmp.npy"
file2_path = "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/annotated_fsim_matrices_complete_livefb/AVA__1107.bmp.npy"

# Load numpy files
file1_data = np.load(file1_path)
file2_data = np.load(file2_path)

# Print shapes
print("Shape of file AVA__1046.bmp.npy:", file1_data.shape)
print("Shape of file AVA__1107.bmp.npy:", file2_data.shape)
