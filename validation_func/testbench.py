from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SequentialSampler
import torch
from math import log10, sqrt
import numpy as np
from skimage import color
from skimage.metrics import structural_similarity as ssim
import cv2

# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/


# def t_PSNR(original, compressed):
#     mse = np.mean((original - compressed) ** 2)
#     if (mse == 0):
#        return 100
#     max_pixel = 255.0
#     psnr = 20 * log10(max_pixel / sqrt(mse))
#     return psnr

def psnr_value(ssi_values_model, img_hr, img_sr):
    return np.append(ssi_values_model, cv2.PSNR(img_hr, img_sr))


def ssi_value(ssi_values_model, img_hr, img_sr):
    gray1 = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_sr, cv2.COLOR_BGR2GRAY)
    return np.append(ssi_values_model, ssim(gray1, gray2, data_range=1))


class Benchmark():
    def __init__(self, deep_aes, device, image_path):
        self.deep_aes = deep_aes
        self.device = device
        self.image_path = image_path
        self.dataloader_hr = None
        self.dataloader_lr = None
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        self.transform_input = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Resize(40),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def load_data(self):
        dataset_hr = ImageFolder(self.image_path, transform=self.transform)
        dataset_lr = ImageFolder(
            self.image_path, transform=self.transform_input)

        rsampler = SequentialSampler(dataset_hr)

        self.dataloader_hr = DataLoader(
            dataset_hr, batch_size=1, shuffle=False, num_workers=8, sampler=rsampler)
        self.dataloader_lr = DataLoader(
            dataset_lr, batch_size=1, shuffle=False, num_workers=8, sampler=rsampler)

    def model_device(self):
        for deep_ae in self.deep_aes:
            deep_ae['dae'].to(self.device)

    def calc(self):
        for deep_ae in self.deep_aes:
            psnr_path_save = f"./np_arr/psnr_{deep_ae['pool_layer']}.dat"
            ssi_path_save = f"./np_arr/ssi_{deep_ae['pool_layer']}.dat"
            ssi_values_mean = np.array([])
            psnr_values_mean = np.array([])
            for t_round in range(1, 4):
                path_to_model = f"./trained_models/layer_{deep_ae['pool_layer']}/{deep_ae['pool_layer']:03}_001_{t_round:03}"
                deep_ae['dae'].load_state_dict(torch.load(path_to_model))
                deep_ae['dae'].eval()
                psnr_values_model = np.array([])
                ssi_values_model = np.array([])
                for batch_hr, batch_lr in zip(self.dataloader_hr, self.dataloader_lr):
                    img_hr, _ = batch_hr
                    img_lr, _ = batch_lr
                    img_lr = img_lr.to(self.device)
                    img_sr = deep_ae['dae'](img_lr)
                    img_hr = img_hr.squeeze().permute(1, 2, 0).numpy()
                    img_sr = img_sr.cpu().detach().squeeze().permute(1, 2, 0).numpy()
                    psnr_values_model = psnr_value(
                        psnr_values_model, img_hr, img_sr)
                    ssi_values_model = ssi_value(
                        ssi_values_model, img_hr, img_sr)

                psnr_values_mean = np.append(
                    psnr_values_mean, np.mean(psnr_values_model))
                ssi_values_mean = np.append(
                    ssi_values_mean, np.mean(ssi_values_model))
                print(f"D-AE: {deep_ae['pool_layer']} --- t_round: {t_round}")

            # psnr_values_mean.tofile(psnr_path_save)
            print(psnr_path_save)
            # ssi_values_mean.tofile(ssi_path_save)

    def run(self):
        self.load_data()
        self.model_device()
        self.calc()
