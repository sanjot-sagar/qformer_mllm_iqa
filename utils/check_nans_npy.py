import os
import numpy as np


def check_npy_files_for_nan(directory, output_directory):
    # Output file to report npy files with NaN values
    output_file = os.path.join(output_directory, 'npy_files_with_nan.txt')
    nan_files_found = False

    with open(output_file, 'w') as report:
        # Traverse the directory
        for filename in os.listdir(directory):
            if filename.endswith('.npy'):
                file_path = os.path.join(directory, filename)
                try:
                    # Load the .npy file
                    data = np.load(file_path)
                    # Check if there are any NaN values
                    if np.isnan(data).any():
                        report.write(f"{filename}\n")
                        print(f"NaN values found in {filename}")
                        nan_files_found = True
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

        # If no NaN values are found, mention that in the report
        if not nan_files_found:
            report.write("No NaN values found in any .npy files.\n")
            print("No NaN values found in any .npy files.")


if __name__ == "__main__":
    # Fixed directory path
    directory = "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/annotated_fsim_matrices"

    # Get the current working directory where the script is executed
    output_directory = os.getcwd()

    if not os.path.isdir(directory):
        print(f"The path {directory} is not a valid directory.")
    else:
        check_npy_files_for_nan(directory, output_directory)
