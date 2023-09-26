import torch
import torch.nn as nn
from torchsummary import summary


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # ------- Encoder ------- #
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu_2 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu_3 = nn.ReLU()
        self.conv_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu_4 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.encoder = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # ------- Decoder ------- #
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu_5 = nn.ReLU()
        self.conv_6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu_6 = nn.ReLU()

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu_7 = nn.ReLU()
        self.conv_8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu_8 = nn.ReLU()

        self.decoder = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.relu_1(x1)
        x2 = self.conv_2(x1)
        x2 = self.relu_2(x2)
        m1 = self.maxpool_1(x2)
        x3 = self.conv_3(m1)
        x3 = self.relu_3(x3)
        x4 = self.conv_4(x3)
        x4 = self.relu_4(x4)
        m2 = self.maxpool_2(x4)
        encoder = self.encoder(m2)

        u1 = self.upsample_1(encoder)
        x5 = self.conv_5(u1)
        x5 = self.relu_5(x5)
        x6 = self.conv_6(x5)
        x6 = self.relu_6(x6)
        add1 = torch.add(x4, x6)
        u2 = self.upsample_2(add1)
        x7 = self.conv_7(u2)
        x7 = self.relu_7(x7)
        x8 = self.conv_8(x7)
        x8 = self.relu_8(x8)
        add2 = torch.add(x2, x8)
        decoder = self.decoder(add2)

        return decoder

    def print_model(self, input_size=(3, 160, 160)):
        summary(self, input_size)
