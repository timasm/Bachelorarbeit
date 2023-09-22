import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim


class Trainer:
    def __init__(self, model, dataset_path, batch_size=64, num_workers=8, learning_rate=0.001):
        self.model = model
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate

        self.transform = transforms.Compose([
            transforms.Resize(260),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

    def load_data(self):
        dataset = ImageFolder(self.dataset_path, transform=self.transform)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return dataloader

    def train(self, num_epochs):
        dataloader = self.load_data()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available and being used")
        else:
            device = torch.device("cpu")
            print("GPU is not available, using CPU instead")

        self.model = self.model.to(device)

        print('Training started.')
        writer = SummaryWriter()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for data in dataloader:
                inputs, _ = data
                inputs = inputs.to(device)

                optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = criterion(outputs, inputs)
                writer.add_scalar("Loss/train", loss, epoch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            # Speichern des Models nach jeder Epoche
            torch.save(self.model.state_dict(),
                       f'trained_model_001_{epoch+1}_{num_epochs}_2109.pth')
            print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.7f}')

        writer.flush()
        writer.close()
        print('Training finished.')
