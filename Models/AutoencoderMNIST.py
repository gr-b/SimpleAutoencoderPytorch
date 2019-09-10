import torch
import torch.nn as nn
import torch.nn.functional as F

bottleneck_size = 10

class AutoencoderMNIST(nn.Module):

    def __init__(self):
        super(AutoencoderMNIST, self).__init__()

        self.create_encoder()
        self.create_decoder()

    def create_encoder(self):
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, bottleneck_size)

    def forward_encoder(self, x):
        x = x.view(-1, 28*28)        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Linear activation
        return x

    def create_decoder(self):
        self.d_fc1 = nn.Linear(bottleneck_size, 64)
        self.d_fc2 = nn.Linear(64, 128)
        self.d_fc3 = nn.Linear(128, 512)
        self.d_fc4 = nn.Linear(512, 784)
        self.d_sig = nn.Sigmoid()

    def forward_decoder(self, z):
        x = F.relu(self.d_fc1(z))
        x = F.relu(self.d_fc2(x))
        x = F.relu(self.d_fc3(x))
        x = F.relu(self.d_fc4(x))
        x = self.d_sig(x)
        return x

    # Forward propagate on input image (or batch of images) x
    def forward(self, x):
       z = self.forward_encoder(x)
       x_prime = self.forward_decoder(z)
       return x_prime


            


