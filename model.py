import torch
import torch.nn as nn


class ESPCNN(nn.Module):
    def __init__(self, num_channels, scale_factor):
        """

        :param num_channels (int): Number of channels in input image
        :param scale_factor (int): Factor to scale-up the input image by
        """

        super(ESPCNN, self).__init__()

        # As per paper, 3 conv layers in backbone, adding padding is optional, not mentioned to use in paper
        # SRCNN paper does not recommend using padding, padding here just helps to visualize the scaled up output image
        # (f1,n1) = (5, 64)
        self.conv1 = nn.Conv2d(in_channels=num_channels, kernel_size=(5, 5), out_channels=64, padding=(2, 2))

        # (f2,n2) = (3, 32)
        self.conv2 = nn.Conv2d(in_channels=64, kernel_size=(3, 3), out_channels=32, padding=(1, 1))

        # f3 = 3, # output shape: H x W x (C x r**2)
        self.conv3 = nn.Conv2d(in_channels=32, kernel_size=(3, 3),
                               out_channels=num_channels * (scale_factor ** 2),
                               padding=(1, 1))
        # Using "Tanh" activation instead of "ReLU"
        self.activation = nn.Tanh()

        # Sub-Pixel Convolution Layer - PixelShuffle
        # rearranges: H x W x (C x r**2) => rH x rW x C
        self.upsampler = nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, inputs):
        # inputs: H x W x C
        x = self.activation(self.conv1(inputs))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        print("shape before up-sampling: ", x.shape)
        out = self.upsampler(x)
        # output: rH x rW x C
        # r: scale_factor
        print("shape after up-sampling: ", out.shape)

        return out


if __name__ == '__main__':
    # Print and Test model outputs with a random input Tensor
    sample_input = torch.rand(size=(1, 3, 224, 224))
    print("Input shape: ", sample_input.shape)

    model = ESPCNN(num_channels=3, scale_factor=2)
    print(f"\n{model}\n")

    # Forward pass with sample input
    output = model(sample_input)
    print(f"output shape: {output.shape}")
