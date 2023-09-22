import sys

from training import train_simple_autoencoder
from test import test_simple_autoencoder


def main():
    # train_simple_autoencoder(num_epochs=20)
    test_simple_autoencoder()


if __name__ == '__main__':
    sys.exit(main())
