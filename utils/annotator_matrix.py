
import os
import numpy
import torch
from torchvision import transforms
from skimage.io import imread
from PIL import Image
import piq
import time
import datetime
from tqdm import tqdm


class AnnotatorDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # List of folders (reference images)
        self.reference_images = sorted(os.listdir(root_dir))
        self.save_path = "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/annotated_fsim_matrices_complete_livefb_1_5"
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.reference_images)

    def __getitem__(self, idx):
        start_time = time.time()
        reference_image_folder = self.reference_images[idx]
        # print(f"Processing images in folder: {reference_image_folder}")
        folder_path = os.path.join(self.root_dir, reference_image_folder)
        all_dists = sorted(os.listdir(folder_path))
        all_images = [os.path.join(folder_path, dist)
                      for dist in all_dists]  # Full path to each image

        images = []
        torch_permuted_images = []
        aligned_param = len(all_images)  # number of spatial fragmentation

        for image_path in all_images:
            curr_image = imread(image_path) / 255.0
            curr_image_torch = torch.from_numpy(curr_image).to(device)
            images.append(curr_image_torch)
            torch_permuted_images.append(curr_image_torch.permute(2, 0, 1))

        # Using aligned_param for matrix size
        annotator_matrix = numpy.zeros((aligned_param, aligned_param))

        for i in range(len(torch_permuted_images)):
            for j in range(0, i):
                im1 = torch_permuted_images[i].unsqueeze(0).to(device)
                im2 = torch_permuted_images[j].unsqueeze(0).to(device)
                annotator_matrix[i][j] = piq.fsim(im1, im2, data_range=1.0)

        annotator_matrix = annotator_matrix + \
            numpy.transpose(annotator_matrix)  # fsim is symmetric
        numpy.fill_diagonal(annotator_matrix, float(1))

        # Save path for annotator matrix
        save_path = os.path.join(
            self.save_path, f"{reference_image_folder}.npy")
        numpy.save(save_path, annotator_matrix)
        # print(f"Annotator matrix saved at: {save_path}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(
        #     f"Time taken for folder {reference_image_folder}: {elapsed_time:.2f} seconds")
        return annotator_matrix


if torch.cuda.is_available():
    # Set device to GPU at index 2
    device = torch.device("cuda:2")
    print("GPU at index 2 is available.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")
# Example usage:
root_dir = "/home/sanjotst/llm_iqa/llm-iqa/datasets/live_fb_synthetic/LIVE_FB_synthetic_full_v2"
dataset = AnnotatorDataset(root_dir)
print('Program started at ' +
      datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
start_time = time.time()
for idx in tqdm(range(len(dataset))):
    # Annotator matrix for each reference image folder will be generated and saved
    annotator_matrix = dataset[idx]
end_time = time.time()
print('Program ended at ' +
      datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
