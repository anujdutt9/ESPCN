import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from glob import glob


def prepare_train_dataset(args):
    h5_file = h5py.File(args.fpath_out, 'w')

    lr_patches = []
    hr_patches = []

    for fpath_image in tqdm(glob(os.path.join(args.dirpath_images, '*.png'))):
        # Load HR image: rH x rW x C, r: scaling factor
        hr_image = cv2.imread(fpath_image).astype(np.float32) / 255.0
        # LR Image: H x W x C
        lr_image = cv2.resize(hr_image,
                              (hr_image.shape[1] // args.scaling_factor, hr_image.shape[0] // args.scaling_factor),
                              interpolation=cv2.INTER_CUBIC)
        assert lr_image.shape[0] == hr_image.shape[0] // args.scaling_factor
        assert lr_image.shape[1] == hr_image.shape[1] // args.scaling_factor

        # Convert BGR to YCbCr
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2YCrCb)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCrCb)

        # As per paper, using only the luminescence channel gave the best outcome
        hr_y = hr_image[:, :, 0]
        lr_y = lr_image[:, :, 0]

        # Get sub-image from Ihr and Ilr as per Sec. 3.2 in paper
        # using patch_size = 17 and stride of 13
        rows = lr_y.shape[0]
        cols = lr_y.shape[1]
        for i in range(0, rows - args.patch_size + 1, args.stride):
            for j in range(0, cols - args.patch_size + 1, args.stride):
                # lr_crop: w = 17, h = 17
                lr_patches.append(lr_y[i:i + args.patch_size, j:j + args.patch_size])
                # hr_crop: w = 17 * r, h = 17 * r
                hr_patches.append(hr_y[i * args.scaling_factor:i * args.scaling_factor + args.patch_size * args.scaling_factor,
                          j * args.scaling_factor:j * args.scaling_factor + args.patch_size * args.scaling_factor])

    lr_patches = np.asarray(lr_patches)
    hr_patches = np.asarray(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()
    print("Done.")


def prepare_val_dataset(args):
    h5_file = h5py.File(args.fpath_out, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, fpath_image in tqdm(enumerate(glob(os.path.join(args.dirpath_images, '*.png')))):
        # Load HR image: rH x rW x C, r: scaling factor
        hr_image = cv2.imread(fpath_image).astype(np.float32) / 255.0
        # LR Image: H x W x C
        lr_image = cv2.resize(hr_image,
                              (hr_image.shape[1] // args.scaling_factor, hr_image.shape[0] // args.scaling_factor),
                              interpolation=cv2.INTER_CUBIC)
        assert lr_image.shape[0] == hr_image.shape[0] // args.scaling_factor
        assert lr_image.shape[1] == hr_image.shape[1] // args.scaling_factor

        # Convert BGR to YCbCr
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2YCrCb)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCrCb)

        # As per paper, using only the luminescence channel gave the best outcome
        hr_y = hr_image[:, :, 0]
        lr_y = lr_image[:, :, 0]

        lr_group.create_dataset(str(i), data=lr_y)
        hr_group.create_dataset(str(i), data=hr_y)

    h5_file.close()
    print("Done.")


def build_parser():
    parser = ArgumentParser(prog="ESPCN Dataset Preparation.")
    parser.add_argument("-i", "--dirpath_images", required=True, type=str,
                        help="Required. Path to dataset directory.")
    parser.add_argument("-o", "--fpath_out", required=True, type=str,
                        help="Required. Path to save dataset files. ex. 'file_name.h5' ")
    parser.add_argument("-ps", "--patch_size", default=17, required=False, type=int,
                        help="Optional. Sub-images patch size.")
    parser.add_argument("-sf", "--scaling_factor", default=2, required=False, type=int,
                        help="Optional. Image Up-scaling factor.")
    parser.add_argument("-s", "--stride", default=13, required=False, type=int,
                        help="Optional. Sub-image extraction stride.")
    parser.add_argument('--val', action='store_true')

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()

    if not args.val:
        print("Creating Training Dataset.")
        prepare_train_dataset(args)
    else:
        print("Creating Validation Dataset.")
        prepare_val_dataset(args)
