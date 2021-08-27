import os
import cv2
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader


# Train Dataset Class
class SRTrainDataset(IterableDataset):
    """
        Training Dataset Class
    """
    def __init__(self, dirpath_images, scaling_factor, patch_size, stride):
        """

        :param dirpath_images (str): path to HR images directory
        :param scaling_factor (int): value by which to scale the HR to LR images
        :param patch_size (int): sub-images size
        :param stride (int): stride for extracting sub-images
        """
        self.dirpath_images = dirpath_images
        self.scaling_factor = scaling_factor
        self.patch_size = patch_size
        self.stride = stride

    def __iter__(self):
        for fpath_image in glob(os.path.join(self.dirpath_images, "*.png")):
            # Load HR image: rH x rW x C, r: scaling factor
            hr_image = cv2.imread(fpath_image).astype(np.float32) / 255.0
            hr_width = (hr_image.shape[1] // self.scaling_factor) * self.scaling_factor
            hr_height = (hr_image.shape[0] // self.scaling_factor) * self.scaling_factor
            hr_image = cv2.resize(hr_image, (hr_width, hr_height), interpolation=cv2.INTER_CUBIC)

            # LR Image: H x W x C
            # As in paper, Sec. 3.2: sub-sample images by up-scaling factor
            lr_image = cv2.resize(hr_image,
                                  (hr_image.shape[1] // self.scaling_factor, hr_image.shape[0] // self.scaling_factor),
                                  interpolation=cv2.INTER_CUBIC) / 255.0
            assert lr_image.shape[0] == hr_image.shape[0] // self.scaling_factor
            assert lr_image.shape[1] == hr_image.shape[1] // self.scaling_factor

            # Convert BGR to YCbCr
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2YCrCb)
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCrCb)

            # As per paper, using only the luminescence channel gave the best outcome
            hr_y = hr_image[:, :, 0]
            lr_y = lr_image[:, :, 0]

            # Get sub-image from Ihr and Ilr as per Sec. 3.2 in paper
            # using patch_size = 17 and stride = 13
            # This ensures that all pixels in the original image appear once and only once as the ground truth of the
            # training data
            rows = lr_y.shape[0]
            cols = lr_y.shape[1]
            for i in range(0, rows - self.patch_size + 1, self.stride):
                for j in range(0, cols - self.patch_size + 1, self.stride):
                    # lr_crop: w = 17, h = 17
                    lr_crop = lr_y[i:i + self.patch_size, j:j + self.patch_size]
                    # hr_crop: w = 17 * r, h = 17 * r
                    hr_crop = hr_y[i * self.scaling_factor:i * self.scaling_factor + self.patch_size * self.scaling_factor, j * self.scaling_factor:j * self.scaling_factor + self.patch_size * self.scaling_factor]
                    lr_crop = np.expand_dims(lr_crop, axis=0)
                    hr_crop = np.expand_dims(hr_crop, axis=0)
                    yield lr_crop, hr_crop

    def __len__(self):
        return len(self.all_images)


# Valid Dataset Class
class SRValidDataset(IterableDataset):
    """
        Validation Dataset Class
    """
    def __init__(self, dirpath_images, scaling_factor):
        """

        :param dirpath_images (str): path to HR validation images
        :param scaling_factor (int): value by which to scale the HR to LR images
        """
        self.dirpath_images = dirpath_images
        self.scaling_factor = scaling_factor

    def __iter__(self):
        for fpath_image in glob(os.path.join(self.dirpath_images, "*.png")):
            # Load HR image: rH x rW x C, r: scaling factor
            hr_image = cv2.imread(fpath_image).astype(np.float32) / 255.0
            hr_width = (hr_image.shape[1] // self.scaling_factor) * self.scaling_factor
            hr_height = (hr_image.shape[0] // self.scaling_factor) * self.scaling_factor
            hr_image = cv2.resize(hr_image, (hr_width, hr_height), interpolation=cv2.INTER_CUBIC)

            # LR Image: H x W x C
            lr_image = cv2.resize(hr_image,
                                  (hr_image.shape[1] // self.scaling_factor, hr_image.shape[0] // self.scaling_factor),
                                  interpolation=cv2.INTER_CUBIC) / 255.0
            assert lr_image.shape[0] == hr_image.shape[0] // self.scaling_factor
            assert lr_image.shape[1] == hr_image.shape[1] // self.scaling_factor

            # Convert BGR to YCbCr
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2YCrCb)
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCrCb)

            # As per paper, using only the luminescence channel gave the best outcome
            hr_y = hr_image[:, :, 0]
            lr_y = lr_image[:, :, 0]

            lr_y = np.expand_dims(lr_y, axis=0)
            hr_y = np.expand_dims(hr_y, axis=0)
            yield lr_y, hr_y

    def __len__(self):
        return len(self.all_images)


# Ref.: https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/6
class ShuffleDataset(IterableDataset):
    """
        Class to Shuffle the Iterable Dataset
    """
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


# Function to return train/val data loader
def get_data_loader(dirpath_train, dirpath_val, scaling_factor, patch_size, stride):
    """ Train/Validation data loader function

    :param dirpath_train (str): path to directory containing high resolution training images
    :param dirpath_val (str): path to directory containing high resolution validation images
    :param scaling_factor (int): Number by which to scale the lr image to hr image
    :param patch_size (int): size of sub-images extracted from original images
    :param stride (int): sub-images extraction stride
    :return (torch.utils.data.DataLoader): training and validation data loader
    """
    # As per paper, Sec. 3.2, sub-images are extracted only during the training phase
    dataset = SRTrainDataset(dirpath_images=dirpath_train,
                            scaling_factor=scaling_factor,
                            patch_size=patch_size,
                            stride=stride)
    train_dataset = ShuffleDataset(dataset, 1024)
    train_loader = DataLoader(train_dataset,
                              batch_size=512,
                              num_workers=8,
                              pin_memory=True)

    valid_dataset = SRValidDataset(dirpath_images=dirpath_val,
                                   scaling_factor=scaling_factor)
    val_loader = DataLoader(valid_dataset,
                            batch_size=1,
                            pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = get_data_loader(dirpath_train="./dataset/DIV2K_train_HR",
                                               dirpath_val="./dataset/DIV2K_valid_HR",
                                               scaling_factor=3,
                                               patch_size=17,
                                               stride=13)

    for idx, (lr_image, hr_image) in enumerate(train_loader):
        print(f"Training - lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        break

    for idx, (lr_image, hr_image) in enumerate(val_loader):
        print(f"Validation - lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        break

    for idx, (lr_image, hr_image) in enumerate(train_loader):
        print(f"lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        lr = lr_image[0].numpy().transpose(1, 2, 0)
        hr = hr_image[0].numpy().transpose(1, 2, 0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(lr)
        ax1.set_title("Low Res")
        ax2.imshow(hr)
        ax2.set_title("High Res")
        plt.show()
        break
