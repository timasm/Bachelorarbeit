import torch
import torch.nn as nn
from torchsummary import summary


class Autoencoder_Deep_3(nn.Module):
    def __init__(self):
        super(Autoencoder_Deep_3, self).__init__()

        # ------- Encoder ------- #
        self.encoder_conv_1 = nn.Conv2d(
            3, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_1 = nn.ReLU()
        self.encoder_conv_2 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_2 = nn.ReLU()
        self.encoder_maxpool_1 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.encoder_conv_3 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_3 = nn.ReLU()
        self.encoder_conv_4 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_4 = nn.ReLU()
        self.encoder_maxpool_2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.enocder_conv_5 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_5 = nn.ReLU()
        self.enocder_conv_6 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_6 = nn.ReLU()
        self.encoder_maxpool_3 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.encoder_conv_7 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_7 = nn.ReLU()

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
            128, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_5 = nn.ReLU()
        self.decoder_conv_6 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_6 = nn.ReLU()

        self.decoder_conv_7 = nn.Conv2d(
            64, 3, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_7 = nn.ReLU()

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
        encoder = self.encoder_relu_7(self.encoder_conv_7(x))

        x = self.decoder_upsample_1(encoder)
        x = self.decoder_relu_1(self.decoder_conv_1(x))
        d2 = self.decoder_relu_2(self.decoder_conv_2(x))
        add = torch.add(d2, e6)
        x = self.decoder_upsample_2(add)
        x = self.decoder_relu_3(self.decoder_conv_3(x))
        d4 = self.decoder_relu_4(self.decoder_conv_4(x))
        add = torch.add(d4, e4)
        x = self.decoder_upsample_3(add)
        x = self.decoder_relu_5(self.decoder_conv_5(x))
        d6 = self.decoder_relu_6(self.decoder_conv_6(x))
        add = torch.add(d6, e2)
        decoder = self.decoder_relu_7(self.decoder_conv_7(add))

        return decoder

    def print_model(self, input_size=(3, 128, 128)):
        summary(self, input_size)
