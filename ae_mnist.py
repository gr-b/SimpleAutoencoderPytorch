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

from Models.AutoencoderMNIST import AutoencoderMNIST as Autoencoder
##################################

batch_size = 256
batch_size_test = 1000 


###################################
#Image loading and preprocessing
###################################

trainLoader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('./data', train=True, download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(
				(0.1307,), (0.3081,)) # Precomputed statistic for normalization
		])),
	batch_size=batch_size, shuffle=True)

testLoader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('./data', train=False, download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(
				(0.1307,), (0.3081,))
		])),
	batch_size=batch_size_test, shuffle=True)


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
		x = x.view(-1, 28*28)		
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
#print(labels)
images = Variable(images).cuda()
reconstructions = model(images)
reconstructions = reconstructions.view(-1, 1, 28, 28)


loss_value = lossFun(reconstructions, images)
print("Loss Value:" + str(loss_value))


# Display images / reconstructions
from matplotlib import pyplot as plt
def show(image):
	plt.imshow(image.permute(1, 2, 0))
	plt.show()

def show2(image1, image2):
	f, axes = plt.subplots(1, 2)
	axes[0].imshow(image1.numpy()[0], cmap='gray')
	axes[1].imshow(image2.numpy()[0], cmap='gray')
	plt.show()

x  = images[0]
x_ = reconstructions[0]

show2(x.cpu(), x_.cpu().detach())






		
	














    



















