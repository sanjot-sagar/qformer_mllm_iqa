# Code for: Dataloading InternLM model
#
# Adapted from: Sanjot's code
# Created Date: 2 April 2024
# Last Modified Date: 16 April 2024
# Last Modified Author: Shika
# modifications : Added synthetic livefb loading code

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from os.path import join
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import h5py
import os


def get_livefb_train_data(config):
    input_csv_file = config.input_csv_file
    df_data = pd.read_csv(input_csv_file).astype({'mos_image': np.float32})
    # Define the train, validation, test split based on 'is_valid' column
    train_indices = df_data.index[df_data['is_valid'] == False].tolist()
    test_indices = df_data.index[df_data['is_valid'] == True].tolist()

    # # randonly pick 5k images from train_indices and test_indices
    # np.random.shuffle(train_indices)
    # train_indices = train_indices[:5000]
    # np.random.shuffle(test_indices)
    # test_indices = test_indices[:5000]

    train_data = df_data.iloc[train_indices]
    test_data = df_data.iloc[test_indices]
    # test_data = test_data[:50]
    return train_data, test_data


def get_livefb_annotation_data(config):
    annotation_directory = config.annotation_directory
    input_csv_file = config.input_csv_file
    df_data = pd.read_csv(input_csv_file).astype({'mos_image': np.float32})
    # get test dataframes
    test_indices = df_data.index[df_data['is_valid'] == True].tolist()
    test_data = df_data.iloc[test_indices]
    # get train dataframes
    train_image_names = [filename.split(
        '.')[0] for filename in os.listdir(annotation_directory)]
    # print train image names to see if .npy is removed properly or not
    train_indices = df_data[df_data['name_image'].apply(
        lambda x: x.split('/')[-1].split('.')[0] in train_image_names)].index
    train_data = df_data.iloc[train_indices]
    return train_data, test_data


class CustomTrainDatasetAnnotatedLIVEFB(Dataset):
    def __init__(self, annotation_matrix_dir, synthetic_img_dir, df_data, model):
        self.annotation_matrix_dir = annotation_matrix_dir
        self.df_data = df_data
        self.synthetic_img_dir = synthetic_img_dir

        if model == "internlm_vl":
            model_image_size = 224
        elif model == "internlm_vl2":
            model_image_size = 490
        elif model == "internlm_vl2_quantised":
            model_image_size = 490
        elif model == "llava":
            model_image_size = 336
        elif model == "mplug_owl":
            model_image_size = 448
        self.transform = transforms.Compose([
            transforms.Resize(
                (model_image_size, model_image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        folder_name = self.df_data.iloc[idx]['name_image']
        # print(folder_name)  # voc_emotic_ava/JPEGImages__2010_005647.jpg
        # error case : blur_dataset/out_of_focus0027.JPG
        folder_name = folder_name.split("/")[-1]
        # print(folder_name)  # JPEGImages__2010_005647.jpg
        # error case : out_of_focus0027.JPG
        if folder_name.endswith(".jpg"):
            folder_name = folder_name.split(".jpg")[0] + ".bmp"
        # print(folder_name)  # JPEGImages__2010_005647.bmp
        if folder_name.endswith(".JPG"):
            folder_name = folder_name.split(".JPG")[0] + ".bmp"

        distorted_images = []
        # Get all images in the folder
        img_paths = sorted(os.listdir(
            join(self.synthetic_img_dir, folder_name)))
        for img_path in img_paths:
            # print(img_path)
            x = Image.open(join(self.synthetic_img_dir, folder_name, img_path))
            if x.mode != 'RGB':
                x = x.convert('RGB')
            x = self.transform(x).float()

            if "REF" in img_path:
                ref_img = x
                # put ref image in front of distorted_images list'
                distorted_images.insert(0, x)
            else:
                distorted_images.append(x)
        distorted_images = torch.stack(distorted_images)
        # Obtain the path to the annotation matrix file
        annotation_matrix_path = join(
            self.annotation_matrix_dir, folder_name + '.npy')
        # Load the annotation matrix
        annotation_matrix = np.load(annotation_matrix_path)
        # print(annotation_matrix.shape)
        # exit()
        np.fill_diagonal(annotation_matrix, 0)
        # Returns 17 images for batch size 1 only
        # Circular shift downward of rows
        annotation_matrix = np.roll(annotation_matrix, 1, axis=0)
        # Circular shift of columns to the right
        annotation_matrix = np.roll(annotation_matrix, 1, axis=1)
        annotation_matrix = torch.from_numpy(annotation_matrix)
        return_sample = {
            "img": distorted_images,
            "mos": torch.tensor(self.df_data.iloc[idx]['mos_image'], dtype=torch.float32),
            "name": self.df_data.iloc[idx]['name_image'],
            "annotation_matrix": annotation_matrix
        }
        return return_sample


class CustomTrainDatasetSyntheticLIVEFB(Dataset):
    def __init__(self, annotation_matrix_dir, synthetic_img_dir, df_data):
        self.synthetic_img_dir = synthetic_img_dir
        self.df_data = df_data
        self.annotation_matrix_dir = annotation_matrix_dir
        print(annotation_matrix_dir)
        # pick 5k
        # self.df_data = self.df_data.sample(n=5000).reset_index(drop=True)

        # while training resize to 256, then random crop to 224 also can be done. Testing normal
        self.transform = transforms.Compose([
            transforms.Resize(
                (224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        folder_name = self.df_data.iloc[idx]['name_image']
        # print(folder_name)  # voc_emotic_ava/JPEGImages__2010_005647.jpg
        # error case : blur_dataset/out_of_focus0027.JPG
        folder_name = folder_name.split("/")[-1]
        # print(folder_name)  # JPEGImages__2010_005647.jpg
        # error case : out_of_focus0027.JPG
        if folder_name.endswith(".jpg"):
            folder_name = folder_name.split(".jpg")[0] + ".bmp"
        # print(folder_name)  # JPEGImages__2010_005647.bmp
        if folder_name.endswith(".JPG"):
            folder_name = folder_name.split(".JPG")[0] + ".bmp"

        distorted_images = []
        # Get all images in the folder
        img_paths = sorted(os.listdir(
            join(self.synthetic_img_dir, folder_name)))
        for img_path in img_paths:
            x = Image.open(join(self.synthetic_img_dir, folder_name, img_path))
            if x.mode != 'RGB':
                x = x.convert('RGB')
            x = self.transform(x).float()

            if "REF" in img_path:
                ref_img = x
                # put ref image in front of distorted_images list'
                distorted_images.insert(0, x)
            else:
                distorted_images.append(x)
        distorted_images = torch.stack(distorted_images)
        # Obtain the path to the annotation matrix file
        annotation_matrix_path = join(
            self.annotation_matrix_dir, folder_name + '.npy')
        # Load the annotation matrix
        annotation_matrix = np.load(annotation_matrix_path)
        np.fill_diagonal(annotation_matrix, 0)
        # Returns 17 images for batch size 1 only
        # Circular shift downward of rows
        annotation_matrix = np.roll(annotation_matrix, 1, axis=0)
        # Circular shift of columns to the right
        annotation_matrix = np.roll(annotation_matrix, 1, axis=1)
        annotation_matrix = torch.from_numpy(annotation_matrix)
        print(annotation_matrix.shape)
        return_sample = {
            "img": distorted_images,
            "mos": torch.tensor(self.df_data.iloc[idx]['mos_image'], dtype=torch.float32),
            "name": self.df_data.iloc[idx]['name_image'],
            "annotation_matrix": annotation_matrix
        }
        return return_sample


class CustomDataset(Dataset):
    def __init__(self, img_dir, df_data, model, data_transform):
        self.img_dir = img_dir
        self.df_data = df_data
        self.data_transform = data_transform
        if model == "internlm_vl":
            model_image_size = 224
            if data_transform == 'normal_transform':
                self.transform = transforms.Compose([
                    transforms.Resize(
                        (model_image_size, model_image_size), interpolation=InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                         (0.26862954, 0.26130258, 0.27577711)),
                ])
            # new iqa relevant transform
            elif data_transform == 'iqa_transform':
                self.transform = transforms.Compose([
                    transforms.Resize(model_image_size,
                                      interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(model_image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                         (0.26862954, 0.26130258, 0.27577711)),
                ])
        elif model == "internlm_vl2":
            model_image_size = 490
            self.transform = transforms.Compose([
                transforms.Resize(
                    (model_image_size, model_image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])
        elif model == "internlm_vl2_quantised":
            model_image_size = 490
        elif model == "llava":
            model_image_size = 336
            self.transform = transforms.Compose([
                transforms.Resize(model_image_size,
                                  interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(model_image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])
        elif model == "mplug_owl":
            model_image_size = 448
            self.transform = transforms.Compose([
                transforms.Resize(model_image_size,
                                  interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(model_image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):

        # if csv, then this
        try:
            filename = join(self.img_dir, self.df_data.iloc[idx]['name_image'])
            mos = torch.tensor(
                self.df_data.iloc[idx]['mos_image'], dtype=torch.float32)
            name = self.df_data.iloc[idx]['name_image']
        # if json, then should be this
        except:
            filename = join(self.img_dir, self.df_data.iloc[idx]['img_path'])
            mos = torch.tensor(
                self.df_data.iloc[idx]['gt_score'], dtype=torch.float32)
            name = self.df_data.iloc[idx]['img_path']

        x = Image.open(filename)
        if x.mode != 'RGB':
            x = x.convert('RGB')
        x = self.transform(x).float()

        return_sample = {
            "img": x,
            "mos": mos,
            "name": name
        }

        return return_sample


class CustomDataset1(Dataset):
    def __init__(self, img_dir, df_data):
        self.img_dir = img_dir
        self.df_data = df_data
        self.transform = transforms.Compose([
            # Resize to 256
            transforms.Resize(
                (256), interpolation=InterpolationMode.BICUBIC),
            # Crop the random 224x224 area, add a random horizontal flip, this is on train set
            # for test set, resize to 224, do a center crop on 224, no horizontal flipping
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):

        # if csv, then this
        try:
            filename = join(self.img_dir, self.df_data.iloc[idx]['name_image'])
            mos = torch.tensor(
                self.df_data.iloc[idx]['mos_image'], dtype=torch.float32)
            name = self.df_data.iloc[idx]['name_image']
        # if json, then should be this
        except:
            filename = join(self.img_dir, self.df_data.iloc[idx]['img_path'])
            mos = torch.tensor(
                self.df_data.iloc[idx]['gt_score'], dtype=torch.float32)
            name = self.df_data.iloc[idx]['img_path']

        x = Image.open(filename)
        if x.mode != 'RGB':
            x = x.convert('RGB')
        x = self.transform(x).float()

        return_sample = {
            "img": x,
            "mos": mos,
            "name": name
        }

        return return_sample


class CustomTrainDatasetSyntheticLIVEFB(Dataset):
    def __init__(self, synthetic_img_dir, df_data):
        self.synthetic_img_dir = synthetic_img_dir
        self.df_data = df_data

        self.transform = transforms.Compose([
            transforms.Resize(
                (224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        folder_name = self.df_data.iloc[idx]['name_image']
        # print(folder_name)  # voc_emotic_ava/JPEGImages__2010_005647.jpg
        # error case : blur_dataset/out_of_focus0027.JPG
        folder_name = folder_name.split("/")[-1]
        # print(folder_name)  # JPEGImages__2010_005647.jpg
        # error case : out_of_focus0027.JPG
        if folder_name.endswith(".jpg"):
            folder_name = folder_name.split(".jpg")[0] + ".bmp"
        # print(folder_name)  # JPEGImages__2010_005647.bmp
        if folder_name.endswith(".JPG"):
            folder_name = folder_name.split(".JPG")[0] + ".bmp"

        distorted_images = []
        # Get all images in the folder
        img_paths = sorted(os.listdir(
            join(self.synthetic_img_dir, folder_name)))

        for img_path in img_paths:
            # print(img_path)
            # printed 5 times
            # JPEGImages__2010_005647_16_01.bmp
            # JPEGImages__2010_005647_16_02.bmp
            # JPEGImages__2010_005647_16_03.bmp
            # JPEGImages__2010_005647_16_04.bmp
            # JPEGImages__2010_005647_16_05.bmp
            # JPEGImages__2010_005647_REF.bmp
            x = Image.open(join(self.synthetic_img_dir, folder_name, img_path))
            if x.mode != 'RGB':
                x = x.convert('RGB')
            x = self.transform(x).float()

            if "REF" in img_path:
                ref_img = x
                # put ref image in front of distorted_images list'
                distorted_images.insert(0, ref_img)
            else:
                distorted_images.append(x)
        distorted_images = torch.stack(distorted_images)

        # Returns 6 images for batch size 1 only
        return_sample = {
            "img": distorted_images,
            # "ref_img": ref_img,
            "mos": torch.tensor(self.df_data.iloc[idx]['mos_image'], dtype=torch.float32),
            "name": self.df_data.iloc[idx]['name_image']
        }
        return return_sample


class CustomTrainDatasetSyntheticLIVEFBAnnotation(Dataset):
    def __init__(self, annotation_matrix_dir, df_data):
        self.annotation_matrix_dir = annotation_matrix_dir
        self.df_data = df_data

        self.transform = transforms.Compose([
            transforms.Resize(
                (224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        folder_name = self.df_data.iloc[idx]['name_image']
        # print(folder_name)  # voc_emotic_ava/JPEGImages__2010_005647.jpg
        # error case : blur_dataset/out_of_focus0027.JPG
        folder_name = folder_name.split("/")[-1]
        # print(folder_name)  # JPEGImages__2010_005647.jpg
        # error case : out_of_focus0027.JPG
        if folder_name.endswith(".jpg"):
            folder_name = folder_name.split(".jpg")[0] + ".bmp"
        # print(folder_name)  # JPEGImages__2010_005647.bmp
        if folder_name.endswith(".JPG"):
            folder_name = folder_name.split(".JPG")[0] + ".bmp"

        distorted_images = []
        # Get all images in the folder
        img_paths = sorted(os.listdir(
            join(self.synthetic_img_dir, folder_name)))

        for img_path in img_paths:
            # print(img_path)
            # printed 5 times
            # JPEGImages__2010_005647_16_01.bmp
            # JPEGImages__2010_005647_16_02.bmp
            # JPEGImages__2010_005647_16_03.bmp
            # JPEGImages__2010_005647_16_04.bmp
            # JPEGImages__2010_005647_16_05.bmp
            # JPEGImages__2010_005647_REF.bmp
            x = Image.open(join(self.synthetic_img_dir, folder_name, img_path))
            if x.mode != 'RGB':
                x = x.convert('RGB')
            x = self.transform(x).float()

            if "REF" in img_path:
                ref_img = x
                # put ref image in front of distorted_images list'
                distorted_images.insert(0, x)
            else:
                distorted_images.append(x)
        distorted_images = torch.stack(distorted_images)

        # Returns 6 images for batch size 1 only
        return_sample = {
            "img": distorted_images,
            # "ref_img": ref_img,
            "mos": torch.tensor(self.df_data.iloc[idx]['mos_image'], dtype=torch.float32),
            "name": self.df_data.iloc[idx]['name_image']
        }
        return return_sample


def get_qinstruct_train_test_loader(config):
    h5_dir = config.h5_dir
    input_json = config.input_json_file

    df_data = pd.read_json(input_json).astype({'gt_score': np.float32})

    # Define the train, validation, test split based on 'is_valid' column
    train_indices = df_data.index[df_data['is_valid'] == False].tolist()
    val_indices = df_data.index[df_data['is_valid'] == True].tolist()

    # Compute test indices based on train and val indices
    overall_indices = set(range(len(df_data)))
    test_indices = list(overall_indices - set(val_indices + train_indices))

    if config.mode == "train":
        # Print indices
        print("\nTrain Indices:", train_indices)
        print("Validation Indices:", val_indices)
        print("Test Indices:", test_indices)

        # Split the dataset based on the indices
        train_set = CustomDatasetQinst(h5_dir, df_data.iloc[train_indices])
        val_set = CustomDatasetQinst(h5_dir, df_data.iloc[val_indices])
        test_set = CustomDatasetQinst(h5_dir, df_data.iloc[test_indices])

        # Print lengths of each set
        print("Train set length:", len(train_set))
        print("Validation set length:", len(val_set))
        print("Test set length:", len(test_set))
        print("Percentage of val data:", (len(val_set)*100) /
              (len(val_set) + len(train_set)))
        print("Percentage of train data:", (len(train_set)*100) /
              (len(val_set) + len(train_set)))

        # Check for common elements between val_indices and train_indices
        intersection = set(val_indices) & set(train_indices)
        if intersection:
            print("There are elements common between val_indices and train_indices.")
            print("Common elements:", intersection)
        else:
            print("There are no common elements between val_indices and train_indices.")

        return train_set, val_set, test_set

    elif config.mode == "eval":
        test_set = CustomDatasetQinst(h5_dir, df_data)
        return test_set


class CustomDatasetQinst(Dataset):
    def __init__(self, h5_dir, df_data):
        self.h5_dir = h5_dir
        self.df_data = df_data

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        h5_file_path = os.path.join(
            self.h5_dir, f"{self.df_data.iloc[idx]['name_patch']}.h5")
        with h5py.File(h5_file_path, 'r') as hf:
            image_name = hf['image_name'][()]
            info_tensor = hf['info_tensor'][()]
            gt_score = hf['gt_score'][()]

        return {
            "info_tensor": torch.tensor(info_tensor, dtype=torch.float32),
            "mos": torch.tensor(gt_score, dtype=torch.float32),
            "name": image_name
        }


def get_test_data(config):
    input_json = config.input_json_file
    df_data = pd.read_json(input_json).astype({'gt_score': np.float32})

    return df_data

# Testing the syntheticlsvq_dataloader
# import argparse
# from torch.utils.data import DataLoader

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img_dir', type=str,
#                         default='/media/shikarao/Data/Shika/01-ilvqa/Databases/LIVEFB/')
#     parser.add_argument('--input_csv_file', type=str,
#                         default='/media/shikarao/Data/Shika/llm-iqa/labels/live_fb_split.csv')
#     parser.add_argument('--synthetic_img_dir', type=str,
#                         default='/media/shikarao/Data/Shika/01-ilvqa/Databases/LIVE_FB_synthetic_full_v2/')
#     config = parser.parse_args()

#     train_data, test_data = get_livefb_train_data(config)
#     print("Train data length:", len(train_data))
#     print("Test data length:", len(test_data))
#     print("Train data:", train_data)
#     print("Test data:", test_data)

#     processed_train_data = CustomTrainDatasetSyntheticLIVEFB(df_data= train_data, synthetic_img_dir= config.synthetic_img_dir)
#     processed_test_data = CustomDataset(df_data= test_data, img_dir= config.img_dir)

#     pooled_train_loader = DataLoader(dataset=processed_train_data, batch_size= 2, shuffle=True)
#     pooled_test_loader = DataLoader(dataset=processed_test_data, batch_size= 1, shuffle=False)

#     for i, sampled_batch in enumerate(pooled_train_loader):
#         dist_img_input = sampled_batch['img']
#         ref_img_input = sampled_batch['ref_img']
#         mos_target = sampled_batch['mos']
