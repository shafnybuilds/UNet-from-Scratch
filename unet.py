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

    def forward(self, img):
        # encoder
        x1 = self.down_conv_1(img)
        print(x1.size())
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        print(x9.size())


if __name__ == "__main__":
    img = torch.rand((1, 1, 572, 572))
    model = UNet()
    print(model(img))
