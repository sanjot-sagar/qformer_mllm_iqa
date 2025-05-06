# has code for the dataloader for the exp2 triplet  loss
# it has custom 5k images

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


def get_livefb_annotation_data(config):
    # what is need is indices of test and train, since train data is reduced to 1/6th of the original, should the test data be also reduced to 1/6th
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


class CustomDataset(Dataset):
    def __init__(self, img_dir, df_data):
        self.img_dir = img_dir
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
