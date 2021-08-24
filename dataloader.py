import os
import cv2
from glob import glob
import albumentations as A
from torch.utils.data import Dataset, DataLoader


# Dataset Class
class SRDataset(Dataset):
    def __init__(self, dirpath_images, scaling_factor):
        self.dirpath_images = dirpath_images
        self.all_images = glob(self.dirpath_images)
        self.scaling_factor = scaling_factor

    def __getitem__(self, item):
        fpath_image = self.all_images[item]
        image = cv2.imread(fpath_image)
        height, width, channels = image.shape
        aug_transform = A.Compose([
            # Blue Ihr using Gaussian Filter
            A.GaussianBlur(),
            # Subsample by scaling factor
            A.Resize(height=height//self.scaling_factor, width=width//self.scaling_factor),
        ])

    def __len__(self):
        return len(self.all_images)


# Function to return train/val data loader
def get_data_loader():
    pass


if __name__ == '__main__':
    pass
