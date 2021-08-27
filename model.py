import torch
import torch.nn as nn


class ESPCN(nn.Module):
    """
        ESPCN Model class
    """
    def __init__(self, num_channels, scale_factor):
        """

        :param num_channels (int): Number of channels in input image
        :param scale_factor (int): Factor to scale-up the input image by
        """

        super(ESPCN, self).__init__()

        # As per paper, 3 conv layers in backbone, adding padding is optional, not mentioned to use in paper
        # SRCNN paper does not recommend using padding, padding here just helps to visualize the scaled up output image
        # Extract input image feature maps
        self.feature_map_layer = nn.Sequential(
            # (f1,n1) = (5, 64)
            nn.Conv2d(in_channels=num_channels, kernel_size=(5, 5), out_channels=64, padding=(2, 2)),
            # Using "Tanh" activation instead of "ReLU"
            nn.Tanh(),
            # (f2,n2) = (3, 32)
            nn.Conv2d(in_channels=64, kernel_size=(3, 3), out_channels=32, padding=(1, 1)),
            # Using "Tanh" activation instead of "ReLU"
            nn.Tanh()
        )

        self.sub_pixel_layer = nn.Sequential(
            # f3 = 3, # output shape: H x W x (C x r**2)
            nn.Conv2d(in_channels=32, kernel_size=(3, 3), out_channels=num_channels * (scale_factor ** 2), padding=(1, 1)),
            # Sub-Pixel Convolution Layer - PixelShuffle
            # rearranges: H x W x (C x r**2) => rH x rW x C
            nn.PixelShuffle(upscale_factor=scale_factor)
        )

    def forward(self, x):
        """ Forward function

        :param x (torch.Tensor): input image
        :return: model output
        """

        # inputs: H x W x C
        x = self.feature_map_layer(x)
        # output: rH x rW x C
        # r: scale_factor
        out = self.sub_pixel_layer(x)

        return out


if __name__ == '__main__':
    # Print and Test model outputs with a random input Tensor
    sample_input = torch.rand(size=(1, 3, 224, 224))
    print("Input shape: ", sample_input.shape)

    model = ESPCN(num_channels=3, scale_factor=3)
    print(f"\n{model}\n")

    # Forward pass with sample input
    output = model(sample_input)
    print(f"output shape: {output.shape}")
