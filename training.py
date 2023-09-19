import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from autoencoder_torch import Autoencoder


def train_simple_autoencoder(num_epochs=1):
    # Train net on GPU
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available and being used")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")

    # Initialize the Autoencoder
    autoencoder = Autoencoder()
    autoencoder = autoencoder.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Load and preprocess your image dataset (you can replace this with your own dataset loading code)
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor()])

    # Download a sample dataset (e.g., CIFAR-10) for demonstration
    dataset = datasets.Caltech101(
        root='./data_2', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # tensorboard
    writer = SummaryWriter()
    # Training loop
    print("Start Training")
    for epoch in range(num_epochs):
        for data in dataloader:
            inputs, _ = data

            inputs = inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = autoencoder(inputs)

            # Calculate the loss
            loss = criterion(outputs, inputs)

            # tensorboard
            writer.add_scalar("Loss/train", loss, epoch)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.7f}')

    # Save the trained model
    torch.save(autoencoder.state_dict(),
               'autoencoder_001_{}_1909.pth'.format(num_epochs))
    writer.flush()
    writer.close()
    print("Training finished and model saved")
