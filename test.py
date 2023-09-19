import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from autoencoder_torch import Autoencoder

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_simple_autoencoder():

    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('autoencoder_001_20_1909.pth'))
    autoencoder.eval()

    # Test the trained autoencoder (you can replace this with your own test images)
    # Load and preprocess your image dataset (you can replace this with your own dataset loading code)
    transform = transforms.Compose([transforms.Resize((480, 480)),
                                    transforms.ToTensor()])
    # Load your test image
    test_image = Image.open('test_image.jpg')
    # Preprocess the test image
    test_image = transform(test_image).unsqueeze(0)
    # Generate the high-resolution image
    output_image = autoencoder(test_image)

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
