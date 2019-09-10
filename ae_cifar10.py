import time
import torch
import torchvision
# Datasets import module

import torchvision.transforms as transforms
# Common image transformations
# Can compose multiple to together and do all at once

import torch.nn as nn
# What is nn vs functional?
from torch.autograd import Variable
from torchvision.utils import save_image

from Models.AutoencoderCIFAR import AutoencoderCIFAR as Autoencoder
##################################

batch_size = 32


###################################
#Image loading and preprocessing
###################################

testTransform = transforms.Compose([ 
    transforms.ToTensor() # Converts an image from [0,255] -> [0,1]
#   , transforms.Normalize(
#            (0.4914, 0.4822, 0.4466), # Mean of each of 3 channels 
#            (0.247,  0.243, 0.261))   # Std of each of 3 channels
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
trainSet = torchvision.datasets.CIFAR10(root='./data',
    train=True, download=True, transform=trainTransform)

testSet = torchvision.datasets.CIFAR10(root='./data', train=False,
    download=True, transform=testTransform)

# For CIFAR10
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 
#            'horse', 'ship', 'truck')


### NOTE: shuffle=False ~~~~~~~~~~~~~~~~~~~
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size,
    shuffle=True, num_workers=4)
testLoader  = torch.utils.data.DataLoader(testSet, batch_size=64,
    shuffle=True, num_workers=4)


##############################
# Training
##############################

num_epochs = 50


minLoss = 50
def checkpoint(loss, model):
	global minLoss
	if loss < minLoss:
		minLoss = loss
		torch.save(model, './checkpoints/model.pt')

model = Autoencoder().cuda()

# We are using a Sigmoid layer at the end so we must use CE loss. Why?
lossFun = nn.MSELoss()#nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)




for epoch in range(num_epochs):
	# TrainLoader is a generator
	start = time.time()
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
	elapsed = time.time() - start
	print('epoch [{}/{}], loss:{:.4f}, time:{:.2f}'.format(epoch+1, num_epochs, loss.data, elapsed))


#######################
# Testing
#######################

checkpoint(loss, model)

images, labels = iter(testLoader).next()
print(labels)
images = Variable(images).cuda()
reconstructions = model(images)

loss_value = lossFun(reconstructions, images)
print("Loss Value:" + str(loss_value))


# Display images / reconstructions
from matplotlib import pyplot as plt
def show(image):
	plt.imshow(image.permute(1, 2, 0))
	plt.show()

def show2(image1, image2):
	f, axes = plt.subplots(1, 2)
	axes[0].imshow(image1.permute(1, 2, 0))
	axes[1].imshow(image2.permute(1, 2, 0))
	plt.show()

x  = images[0]
x_ = reconstructions[0]

show2(x.cpu(), x_.cpu().detach())






		
	














    



















