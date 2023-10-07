import torch
import torch.nn as nn
from torchsummary import summary


class Autoencoder_Sigmoid(nn.Module):
    def __init__(self):
        super(Autoencoder_Sigmoid, self).__init__()

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

        self.encoder_conv_5 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, padding_mode='replicate')
        self.encoder_relu_5 = nn.ReLU()

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
            128, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_3 = nn.ReLU()
        self.decoder_conv_4 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.decoder_relu_4 = nn.ReLU()

        self.decoder_conv_5 = nn.Conv2d(64, 3, kernel_size=3,
                                        padding=1, padding_mode='replicate')
        self.decoder_relu_5 = nn.ReLU()
        self.decoder_sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.encoder_relu_1(self.encoder_conv_1(x))
        e2 = self.encoder_relu_2(self.encoder_conv_2(e1))
        m1 = self.encoder_maxpool_1(e2)
        e3 = self.encoder_relu_3(self.encoder_conv_3(m1))
        e4 = self.encoder_relu_4(self.encoder_conv_4(e3))
        m2 = self.encoder_maxpool_2(e4)
        encoder = self.encoder_relu_5(self.encoder_conv_5(m2))

        u1 = self.decoder_upsample_1(encoder)
        d1 = self.decoder_relu_1(self.decoder_conv_1(u1))
        d2 = self.decoder_relu_2(self.decoder_conv_2(d1))
        add1 = torch.add(d2, e4)
        u2 = self.decoder_upsample_2(add1)
        d3 = self.decoder_relu_3(self.decoder_conv_3(u2))
        d4 = self.decoder_relu_4(self.decoder_conv_4(d3))
        add2 = torch.add(d4, e2)
        decoder = self.decoder_relu_5(self.decoder_conv_5(add2))
        decoder = self.decoder_sigmoid(decoder)

        return decoder

    def print_model(self, input_size=(3, 128, 128)):
        summary(self, input_size)
