import torch
import torchvision as tv
# Datasets import module

import torchvision.transforms as transforms
# Common image transformations
# Can compose multiple to together and do all at once

import torch.nn as nn
# What is nn vs functional?
from torch.autograd import Variable
from torchvision.utils import save_image

from Models.Autoencoder import Autoencoder




###################################
#Image loading and preprocessing
###################################

testTransform = transforms.Compose([ 
    transforms.ToTensor(), # Converts an image from [0,255] -> [0,1]
    transforms.Normalize(
            (0.4914, 0.4822, 0.4466), # Mean of each of 3 channels 
            (0.247,  0.243, 0.261))   # Std of each of 3 channels
])  # ------> These means, stds are just precomputed I guess


trainTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4466), 
        (0.247, 0.243, 0.261))
    ])
# ^^ This one just uses torchvision transforms instead of 



###
# Load Dataset
###
trainSet = tv.datasets.CIFAR10(root='./data',
    train=True, download=True, transform=trainTransform)

testSet = tv.datasets.CIFAR10(root='./data', train=False,
    download=True, transform=testTransform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 
            'horse', 'ship', 'truck')


### NOTE: shuffle=False ~~~~~~~~~~~~~~~~~~~
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=32,
    shuffle=False, num_workers=4)
testLoader  = torch.utils.data.DataLoader(testSet, batch_size=32,
    shuffle=False, num_workers=4)


##############################
# Training
##############################

num_epochs = 5
batch_size = 128

model = Autoencoder().cuda()

# We are using a Sigmoid layer at the end so we must use CE loss. Why?
lossFun = nn.MSELoss()#nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)


for epoch in range(num_epochs):
	# TrainLoader is a generator
	for data in trainLoader:		
		x, _ = data # Each 'data' is an image, label pair
		x = Variable(x).cuda() # Input image must be a tensor and moved to the GPU			

		# Forward pass
		x_prime = model(x) # pass through into a reconstructed image
		loss = lossFun(x_prime, x)

		# Backward pass
		optimizer.zero_grad() # Backward function accumulates gradients, so we don't want to mix up gradients. 
				      # Set to zero instead.
		loss.backward()
		optimizer.step()
	print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data))



		
	














    



















