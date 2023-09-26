import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
import torchvision.transforms as transforms
import torch.optim as optim


class Trainer:
    def __init__(self, model, device, dataset_path, batch_size=32, num_workers=8, learning_rate=0.001, l1_lambda=0.00001):
        self.model = model
        self.device = device
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ])

        self.transform_input = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.Resize(40),
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])

    def load_data(self):
        dataset = ImageFolder(self.dataset_path, transform=self.transform)
        print(len(dataset))
        dataset_input = ImageFolder(
            self.dataset_path, transform=self.transform_input)

        indices = torch.randperm(len(dataset))[:20000]
        rsampler = SubsetRandomSampler(indices)
        # rsampler = SequentialSampler(dataset)

        dataloader_original = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=rsampler)
        dataloader_downsampled = DataLoader(
            dataset_input, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=rsampler)

        return dataloader_original, dataloader_downsampled

    def train(self, num_epochs):
        dataloader_original, dataloader_downsampled = self.load_data()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print('Training started.')
        writer = SummaryWriter()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch_original, batch_downsampled in zip(dataloader_original, dataloader_downsampled):
                o_images, _ = batch_original
                d_images, _ = batch_downsampled

                original_images = o_images.to(self.device)
                inputs = d_images.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, original_images)
                l1_reg = 0
                for param in self.model.parameters():
                    l1_reg += torch.abs(param).sum()
                loss += self.l1_lambda * l1_reg
                writer.add_scalar("Loss/train", loss, epoch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader_original)
            torch.save(self.model.state_dict(),
                       f'trained_model_003_{(epoch+1):03}_{loss.item():.7f}.pth')
            print(
                f'Epoch [{epoch + 1}/{num_epochs}] Avg Loss: {avg_loss:.7f} Curr Loss: {loss.item():.7f}')

        writer.flush()
        writer.close()
        print('Training finished.')
