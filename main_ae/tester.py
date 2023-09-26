from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Tester:
    def __init__(self, model, input_image_path, output_image_path):
        self.model = model
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.transform_input = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.Resize(60),
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])

    def generate_super_resolution_image(self):
        input_image = Image.open(self.input_image_path)
        input_image = self.transform_input(input_image).unsqueeze(0)

        super_resolution_image = self.model(input_image)
        super_resolution_image = super_resolution_image.squeeze(0)
        super_resolution_image = transforms.ToPILImage()(super_resolution_image)
        super_resolution_image.save(self.output_image_path)

    def show_super_resolution_image(self):
        input_image = Image.open(self.input_image_path)
        input_image = self.transform_input(input_image).unsqueeze(0)
        super_resolution_image = self.model(input_image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(input_image.squeeze().permute(1, 2, 0))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Generated Image')
        plt.imshow(super_resolution_image.squeeze().detach().permute(1, 2, 0))
        plt.axis('off')
        plt.show()
