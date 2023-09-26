from model import Autoencoder
from trainer import Trainer
from tester import Tester
import torch


def main():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available and being used")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")

    modeTrain = False
    autoencoder = Autoencoder()
    if modeTrain:
        autoencoder.to(device)
        autoencoder.print_model()
        trainer = Trainer(autoencoder, device, "256_ObjectCategories")
        #                  "img/lfw-deepfunneled/lfw-deepfunneled")
        trainer.train(num_epochs=40)
    else:
        model_path = 'trained_model/003/trained_model_003_020_0.0041530.pth'
        autoencoder.load_state_dict(torch.load(model_path))
        autoencoder.eval()
        tester = Tester(autoencoder, input_image_path="Ross_Verba_0001.jpg",
                        output_image_path="test_out.jpg")
        tester.show_super_resolution_image()
        # tester.generate_super_resolution_image()


if __name__ == "__main__":
    main()
