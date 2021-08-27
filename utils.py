import torch


class AverageMeter(object):
    """ Function to Compute and store the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_psnr(img1, img2):
    """ Function to calculate the Peak Signal to Noise Ratio (PSNR)

    :param img1: model output Isr image
    :param img2: ground truth Ihr image
    :return: Peak Signal to Noise Ratio between two images
    """

    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))