import torch
import torch.nn as nn


def double_conv(input_channel, output_channel):
    conv = nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channel, output_channel, kernel_size=3),
        nn.ReLU(inplace=True),
    )

    return conv


# func to crop the image to concatenate with the up_conv image
def crop_image(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta : tensor_size - delta, delta : tensor_size - delta]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # max pool 2x2
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        # second part Transpose Conv initialization
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up_conv_1 = double_conv(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_conv_2 = double_conv(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_conv_3 = double_conv(256, 128)

        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_conv_4 = double_conv(128, 64)

        # output layer
        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, img):
        # encoder
        x1 = self.down_conv_1(
            img
        )  # this ouput should be passed to second part of the UNet, I mean the up conv process, it should be cropped and matched the size of the image in that phase
        # i marked the # as the refernce
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)  #
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)  #
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)  #
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        # decode
        x = self.up_transpose_1(x9)
        y = crop_image(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_transpose_2(x)
        y = crop_image(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_transpose_3(x)
        y = crop_image(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.up_transpose_4(x)
        y = crop_image(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))

        # output channel
        x = self.out(x)
        print(x.size())
        return x


if __name__ == "__main__":
    img = torch.rand((1, 1, 572, 572))
    model = UNet()
    print(model(img))
