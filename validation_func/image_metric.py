from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SequentialSampler
import torch
from ignite.metrics import PSNR
from ignite.engine import *
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt


def calculate_psnr(tensor1, tensor2):
    mse = torch.mean((tensor1 - tensor2) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def psnr_value(psnr_values_model, img_hr, img_sr):
    return np.append(psnr_values_model, calculate_psnr(img_hr, img_sr))


def ssi_value(ssi_values_model, img_hr, img_sr):
    gray1 = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_sr, cv2.COLOR_BGR2GRAY)
    return np.append(ssi_values_model, ssim(gray1, gray2, data_range=1))


class Benchmark():
    def __init__(self, aes, device, image_path):
        self.aes = aes
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
        for ae in self.aes:
            ae['ae'].to(self.device)

    def calc(self):
        for ae in self.aes:
            ssi_values_mean = np.array([])
            psnr_values_mean = np.array([])
            for t_round in range(0, 60):
                path_to_trained_model = f"./{ae['train_path']}/{(t_round+1):03}"
                ae['ae'].load_state_dict(torch.load(path_to_trained_model))
                ae['ae'].eval()
                psnr_values_model = np.array([])
                ssi_values_model = np.array([])
                for batch_hr, batch_lr in zip(self.dataloader_hr, self.dataloader_lr):
                    img_hr, _ = batch_hr
                    img_lr, _ = batch_lr
                    img_hr = img_hr.to(self.device)
                    img_lr = img_lr.to(self.device)
                    img_sr = ae['ae'](img_lr)

                    psnr_values_model = psnr_value(
                        psnr_values_model, img_hr, img_sr)

                    img_hr = img_hr.cpu().detach().squeeze().permute(1, 2, 0).numpy()
                    img_sr = img_sr.cpu().detach().squeeze().permute(1, 2, 0).numpy()
                    ssi_values_model = ssi_value(
                        ssi_values_model, img_hr, img_sr)

                psnr_values_mean = np.append(
                    psnr_values_mean, np.mean(psnr_values_model))
                ssi_values_mean = np.append(
                    ssi_values_mean, np.mean(ssi_values_model))
                print(f"ae: {ae['name']} --- t_round: {t_round+1}")

            psnr_values_mean.tofile(ae['psnr_path'])
            ssi_values_mean.tofile(ae['ssim_path'])

    def plot_psnr(self):
        arr = []
        for ae in self.aes:
            arr.append(np.fromfile(ae['psnr_path']))
        x = np.linspace(7, 60, 53)
        plt.figure(figsize=(8, 6))
        for data, ae in zip(arr, self.aes):
            print(f"{ae['name']} - PSNR: {data[59]}")
            plt.plot(x, data[7:], label=f"{ae['name']}")
        plt.xlabel('Trainings Runde')
        plt.ylabel('dB')
        plt.legend()
        plt.title('PSNR')
        plt.grid(True)
        plt.show()

    def plot_ssim(self):
        arr = []
        for ae in self.aes:
            arr.append(np.fromfile(ae['ssim_path']))
        x = np.linspace(7, 60, 53)
        plt.figure(figsize=(8, 6))
        for data, ae in zip(arr, self.aes):
            print(f"{ae['name']} - SSIM: {data[59]}")
            plt.plot(x, data[7:], label=f"{ae['name']}")
        plt.xlabel('Trainings Runde')
        plt.ylabel('')
        plt.legend()
        plt.title('ssim')
        plt.grid(True)
        plt.show()

    def run(self):
        self.load_data()
        self.model_device()
        self.calc()
