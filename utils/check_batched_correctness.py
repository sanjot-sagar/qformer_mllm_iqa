import os
import numpy as np


def compare_npy_files(dir1, dir2):
    if not os.path.isdir(dir1) or not os.path.isdir(dir2):
        print("Error: One or both directories do not exist.")
        return

    files1 = sorted(os.listdir(dir1))
    files2 = sorted(os.listdir(dir2))

    if len(files1) != len(files2):
        print(
            f"Directories {dir1} and {dir2} do not have the same number of files.")
        return

    unequal_files = []

    for file_name in files1:
        file_path1 = os.path.join(dir1, file_name)
        file_path2 = os.path.join(dir2, file_name)

        if not os.path.isfile(file_path1) or not os.path.isfile(file_path2):
            print(f"File {file_name} is missing in one of the directories.")
            continue

        data1 = np.load(file_path1)
        data2 = np.load(file_path2)

        if not np.array_equal(data1, data2):
            print(f"Files {file_path1} and {file_path2} are not equal.")
            unequal_files.append(file_name)
        else:
            print(f"Files {file_path1} and {file_path2} are equal.")

    if not unequal_files:
        print("All files are equal.")
    else:
        print("The following files are not equal:")
        for file in unequal_files:
            print(file)


# Specify your directories
dir1 = "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/annotator_matrix_testing/annotator_matrices_batch"
dir2 = "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/annotator_matrix_testing/annotator_matrices_non_batch"

compare_npy_files(dir1, dir2)
