import torch
import torch.nn as nn
import torch.nn.functional as F



class AutoencoderMNIST(nn.Module):

    def __init__(self):
        super(AutoencoderMNIST, self).__init__()

        self.create_encoder()
        self.create_decoder()

    def create_encoder(self):
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3) # Out: (10x26x26)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3) # Out: (20x24x24)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward_encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) # Add maxpool2d layers?
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def create_decoder(self):
        self.d_fc1 = nn.Linear(10, 50)
        self.d_fc2 = nn.Linear(50, 320)
        self.d_convt1 = nn.ConvTranspose2d(20, 10, kernel_size=3)
        self.d_convt2 = nn.ConvTranspose2d(10, 1, kernel_size=3)
        self.d_sig   = nn.Sigmoid()

    def forward_decoder(self, z):
        x = F.relu(self.d_fc1(z))
        x = F.relu(self.d_fc2(x))
        x = x.view(-1, 20, 24, 24)
        x = F.relu(self.d_convt1(x))
        x = F.relu(self.d_convt2(x))
        x = self.d_sig(x)
        return x

    # Forward propagate on input image (or batch of images) x
    def forward(self, x):
       z = self.forward_encoder(x)
       x_prime = self.forward_decoder(z)
       return x_prime


            


