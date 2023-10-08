import torch
import torch.nn as nn
from torchsummary import summary


class Autoencoder_Deep_1(nn.Module):
    def __init__(self):
        super(Autoencoder_Deep_1, self).__init__()

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

        # ------- Decoder ------- #
        self.decoder_upsample_1 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.decoder_conv_1 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_1 = nn.ReLU()
        self.decoder_conv_2 = nn.Conv2d(
            128, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_2 = nn.ReLU()

        self.decoder_conv_3 = nn.Conv2d(
            64, 3, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_3 = nn.ReLU()

    def forward(self, x):
        x = self.encoder_relu_1(self.encoder_conv_1(x))
        e2 = self.encoder_relu_2(self.encoder_conv_2(x))
        x = self.encoder_maxpool_1(e2)
        encoder = self.encoder_relu_3(self.encoder_conv_3(x))

        x = self.decoder_upsample_1(encoder)
        x = self.decoder_relu_1(self.decoder_conv_1(x))
        d2 = self.decoder_relu_2(self.decoder_conv_2(x))
        add = torch.add(d2, e2)
        decoder = self.decoder_relu_3(self.decoder_conv_3(add))

        return decoder

    def print_model(self, input_size=(3, 128, 128)):
        summary(self, input_size)
