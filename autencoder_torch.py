import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        
        # Bottleneck layer
        self.bottleneck = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

# Initialize the Autoencoder
autoencoder = Autoencoder()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Load and preprocess your image dataset (you can replace this with your own dataset loading code)
transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor()])

# Download a sample dataset (e.g., CIFAR-10) for demonstration
dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(epoch)
    for data in dataloader:
        inputs, _ = data
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = autoencoder(inputs)
        
        # Calculate the loss
        loss = criterion(outputs, inputs)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}')

# Save the trained model
torch.save(autoencoder.state_dict(), 'autoencoder.pth')

# Test the trained autoencoder (you can replace this with your own test images)
test_image = Image.open('test_image.jpg')  # Load your test image
test_image = transform(test_image).unsqueeze(0)  # Preprocess the test image
output_image = autoencoder(test_image)  # Generate the high-resolution image

# Display the original and generated images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(test_image.squeeze().permute(1, 2, 0))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Generated Image')
plt.imshow(output_image.squeeze().detach().permute(1, 2, 0))
plt.axis('off')

plt.show()
