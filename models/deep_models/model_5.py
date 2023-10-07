import torch
import torch.nn as nn
from torchsummary import summary


class Autoencoder_Deep_5(nn.Module):
    def __init__(self):
        super(Autoencoder_Deep_5, self).__init__()

        # ------- Encoder ------- #
        self.encoder_conv_1 = nn.Conv2d(
            3, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_1 = nn.ReLU()
        self.encoder_conv_2 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_2 = nn.ReLU()
        self.encoder_maxpool_1 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.encoder_conv_3 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_3 = nn.ReLU()
        self.encoder_conv_4 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_4 = nn.ReLU()
        self.encoder_maxpool_2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.enocder_conv_5 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_5 = nn.ReLU()
        self.enocder_conv_6 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_6 = nn.ReLU()
        self.encoder_maxpool_3 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.enocder_conv_7 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_7 = nn.ReLU()
        self.enocder_conv_8 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_8 = nn.ReLU()
        self.encoder_maxpool_4 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.encoder_conv_9 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_9 = nn.ReLU()
        self.encoder_conv_10 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_10 = nn.ReLU()
        self.encoder_maxpool_5 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.encoder_conv_11 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_11 = nn.ReLU()

        # ------- Decoder ------- #
        self.decoder_upsample_1 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.decoder_conv_1 = nn.Conv2d(
            256, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_1 = nn.ReLU()
        self.decoder_conv_2 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_2 = nn.ReLU()

        self.decoder_upsample_2 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.decoder_conv_3 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_3 = nn.ReLU()
        self.decoder_conv_4 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_4 = nn.ReLU()

        self.decoder_upsample_3 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.decoder_conv_5 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_5 = nn.ReLU()
        self.decoder_conv_6 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_6 = nn.ReLU()

        self.decoder_upsample_4 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.decoder_conv_7 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_7 = nn.ReLU()
        self.decoder_conv_8 = nn.Conv2d(
            128, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_8 = nn.ReLU()

        self.decoder_upsample_5 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.decoder_conv_9 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_9 = nn.ReLU()
        self.decoder_conv_10 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_10 = nn.ReLU()

        self.decoder_conv_11 = nn.Conv2d(
            64, 3, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_11 = nn.ReLU()

    def forward(self, x):
        x = self.encoder_relu_1(self.encoder_conv_1(x))
        e2 = self.encoder_relu_2(self.encoder_conv_2(x))
        x = self.encoder_maxpool_1(e2)
        x = self.encoder_relu_3(self.encoder_conv_3(x))
        e4 = self.encoder_relu_4(self.encoder_conv_4(x))
        x = self.encoder_maxpool_2(e4)
        x = self.encoder_relu_5(self.enocder_conv_5(x))
        e6 = self.encoder_relu_6(self.enocder_conv_6(x))
        x = self.encoder_maxpool_3(e6)
        x = self.encoder_relu_7(self.enocder_conv_7(x))
        e8 = self.encoder_relu_8(self.enocder_conv_8(x))
        x = self.encoder_maxpool_4(e8)
        x = self.encoder_relu_9(self.encoder_conv_9(x))
        e10 = self.encoder_relu_10(self.encoder_conv_10(x))
        x = self.encoder_maxpool_5(e10)
        encoder = self.encoder_relu_11(self.encoder_conv_11(x))

        x = self.decoder_upsample_1(encoder)
        x = self.decoder_relu_1(self.decoder_conv_1(x))
        d2 = self.decoder_relu_2(self.decoder_conv_2(x))
        add = torch.add(d2, e10)
        x = self.decoder_upsample_2(add)
        x = self.decoder_relu_3(self.decoder_conv_3(x))
        d4 = self.decoder_relu_4(self.decoder_conv_4(x))
        add = torch.add(d4, e8)
        x = self.decoder_upsample_3(add)
        x = self.decoder_relu_5(self.decoder_conv_5(x))
        d6 = self.decoder_relu_6(self.decoder_conv_6(x))
        add = torch.add(d6, e6)
        x = self.decoder_upsample_4(add)
        x = self.decoder_relu_7(self.decoder_conv_7(x))
        d8 = self.decoder_relu_8(self.decoder_conv_8(x))
        add = torch.add(d8, e4)
        x = self.decoder_upsample_5(add)
        x = self.decoder_relu_9(self.decoder_conv_9(x))
        d10 = self.decoder_relu_10(self.decoder_conv_10(x))
        add = torch.add(d10, e2)
        decoder = self.decoder_relu_11(self.decoder_conv_11(add))

        return decoder

    def print_model(self, input_size=(3, 128, 128)):
        summary(self, input_size)
