from os.path import join
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, img_dir, df_data, transform=transforms.ToTensor(), num_aug=1, num_samples=0):
        self.img_dir = img_dir
        self.df_data = df_data
        self.transform = transform

        # Truncate dataset while testing
        if num_samples:
            self.df_data = self.df_data[:num_samples]
        # Repeat the data points for showing multiple augmentations (defined in transform) from the same image
        if num_aug != 1:
            self.df_data = self.df_data.loc[self.df_data.index.repeat(
                num_aug)].reset_index(drop=True)

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        x = Image.open(join(self.img_dir, self.df_data.iloc[idx]['img_path']))
        if x.mode != 'RGB':
            x = x.convert('RGB')
        x = self.transform(x)
        return x, self.df_data.iloc[idx]['gt_score']
