from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Tester:
    def __init__(self, model, px, input_image_path, output_image_path):
        self.model = model
        self.px = px
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.transform_input = transforms.Compose([
            transforms.Resize(self.px),
            transforms.Resize(40),
            transforms.Resize(self.px),
            transforms.ToTensor()
        ])
        self.transform_hr = transforms.Compose([
            transforms.Resize(self.px)
        ])

    def generate_super_resolution_image(self):
        input_image = Image.open(self.input_image_path)
        input_image = self.transform_input(input_image).unsqueeze(0)

        super_resolution_image = self.model(input_image)
        super_resolution_image = super_resolution_image.squeeze(0)
        super_resolution_image = transforms.ToPILImage()(super_resolution_image)
        super_resolution_image.save(self.output_image_path)

    def show_super_resolution_image(self):
        image = Image.open(self.input_image_path)
        hr_image = self.transform_hr(image)
        lr_image = self.transform_input(image).unsqueeze(0)
        sr_image = self.model(lr_image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title('HR Image')
        plt.imshow(hr_image)
        plt.axis('off')

        # Subplot 2: LR Image
        plt.subplot(1, 3, 2)
        plt.title('LR Image')
        plt.imshow(lr_image.squeeze().permute(1, 2, 0))
        plt.axis('off')

        # Subplot 3: SR Image
        plt.subplot(1, 3, 3)
        plt.title('SR Image')
        plt.imshow(sr_image.squeeze().detach().permute(1, 2, 0))
        plt.axis('off')
        plt.show()
