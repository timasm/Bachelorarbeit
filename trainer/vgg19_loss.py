import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.vgg_feature_extractor = vgg_feature_extractor
        self.criterion = nn.MSELoss()

    def forward(self, output_images, target_images):
        output_features = self.vgg_feature_extractor(output_images)
        target_features = self.vgg_feature_extractor(target_images)
        loss = 0.0
        for out_feat, target_feat in zip(output_features, target_features):
            loss += self.criterion(out_feat, target_feat)
        return loss


class Trainer_mse_perceputal:
    def __init__(self, model, device, px, perceputal_loss, dataset_path, trainings_round=0, batch_size=32, num_workers=8, learning_rate=0.0001, l1_lambda=0.00001):
        self.model = model
        self.device = device
        self.px = px
        self.perceputal_loss = perceputal_loss
        self.dataset_path = dataset_path
        self.trainings_round = trainings_round
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.transform = transforms.Compose([
            transforms.Resize(self.px),
            transforms.ToTensor(),
        ])
        self.transform_input = transforms.Compose([
            transforms.Resize(self.px),
            transforms.Resize(40),
            transforms.Resize(self.px),
            transforms.ToTensor()
        ])

    def load_data(self):
        dataset = ImageFolder(self.dataset_path, transform=self.transform)
        dataset_input = ImageFolder(
            self.dataset_path, transform=self.transform_input)
        rsampler = SequentialSampler(dataset)

        dataloader_original = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=rsampler)
        dataloader_downsampled = DataLoader(
            dataset_input, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=rsampler)

        return dataloader_original, dataloader_downsampled

    def train(self, num_epochs):
        dataloader_original, dataloader_downsampled = self.load_data()

        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        vgg19_features = vgg19.features.to(self.device).eval()
        perceptual_loss = PerceptualLoss(vgg19_features)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate, weight_decay=1e-5)

        print('Training started.')
        writer = SummaryWriter()
        for epoch in range(num_epochs):
            total_loss = 0.0
            mse_loss = 0.0
            combined_loss = 0.0
            for batch_original, batch_downsampled in zip(dataloader_original, dataloader_downsampled):
                o_images, _ = batch_original
                d_images, _ = batch_downsampled

                original_images = o_images.to(self.device)
                inputs = d_images.to(self.device)

                outputs = self.model(inputs)
                mse_loss = criterion(outputs, original_images)

                l1_reg = 0
                for param in self.model.parameters():
                    l1_reg += torch.abs(param).sum()
                mse_loss += self.l1_lambda * l1_reg

                perceptual_loss_value = perceptual_loss(
                    outputs, original_images)
                combined_loss = (1-self.perceputal_loss)*mse_loss + \
                    self.perceputal_loss*perceptual_loss_value

                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
                writer.add_scalar("MSE-Loss/train", mse_loss, epoch)
                writer.add_scalar("Perceputal-Loss/train",
                                  perceptual_loss_value, epoch)
                writer.add_scalar("Combined-Loss/train", combined_loss, epoch)

                total_loss += combined_loss.item()

            torch.save(self.model.state_dict(),
                       f'./trained_models/perceptual_loss/000_100/{(epoch+1+self.trainings_round):03}')
            print(
                f'Epoch [{(epoch+1+self.trainings_round):03}/{num_epochs + self.trainings_round}] -- Per: {100*self.perceputal_loss} -- MSE Loss: {mse_loss.item():.7f} -- Perceptual Loss: {perceptual_loss_value.item():.7f} -- Curr Loss: {combined_loss.item():.7f}')

        writer.flush()
        writer.close()
        print('Training finished.')
