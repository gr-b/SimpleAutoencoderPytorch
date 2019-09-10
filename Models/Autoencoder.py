import torch
import torch.nn as nn
import torch.nn.functional as F



class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = self.createEncoder()
        self.decoder = self.createDecoder()

    def createEncoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(5, 5)), 
            # 3 channels in (RGB), 6 channels out (so 6 kernels)
            # Shape of output: (n-(k-1)) = 32-(5-1) = 32-4 = 28
                                # (6x28x28)
            nn.ReLU(True), # True -> Do the operation in place (save memory)
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(True))
        # The bottleneck latent dimension is (6x24x24) = 9216 
        # Vs the input image size of (3x32x32) = 3072

    def createDecoder(self):
        return nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=(5,5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=(5, 5)),
            nn.ReLU(True),
            nn.Sigmoid())

    # Forward propagate on input image (or batch of images) x
    def forward(self, x):
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return x_prime


            


