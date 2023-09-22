from model import Autoencoder
from trainer import Trainer
from PIL import Image
import os


def transform_images(dataset_path, dataset_path_resized):
    os.makedirs(dataset_path_resized, exist_ok=True)

    # Durchlaufen Sie den Caltech-256-Datensatz und schneiden der Bilder auf 256x256 Pixel
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                output_file_path = os.path.join(dataset_path_resized, file)
                with Image.open(file_path) as img:
                    img = img.resize((256, 256), Image.ANTIALIAS)
                    img.save(output_file_path)


def main():
    # dataset_path = 'caltech256'
    path = 'caltech256'

    # transform_images(dataset_path, dataset_path_resized)

    autoencoder = Autoencoder()
    print(autoencoder)
    trainer = Trainer(autoencoder, path)
    trainer.train(num_epochs=6)


if __name__ == "__main__":
    main()
