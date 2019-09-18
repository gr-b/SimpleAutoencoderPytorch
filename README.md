# SimpleAutoencoder - Pytorch implementation
To run: `python3 ae_mnist.py`
To run the interactive latent space exploration, create a folder called `checkpoints` and
train the model once using `python3 ae_mnist_interactive_latent.py`.
Next, run it a second time. The second run will pick up the model in your checkpoints folder and use it to
show the visualization.


# Notes on problems encountered
1. Having Sigmoid(Relu(x)) as the last part of the decoder creates a very odd latent space
2. Using normalization about the mean and std of the MNIST dataset results in the output of the decoder being constrained to be either 0.5 or 1, making it very hard to train.
