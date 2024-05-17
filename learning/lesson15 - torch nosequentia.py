# Simple multi layer perceptron being used for classification

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torchmetrics
import matplotlib.pyplot as plt

# load the training and the test datasets
training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

# img, label = training_data[25]
# plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# training_data[10] -> to see one of the images from the dataset
device = "cuda" if torch.cuda.is_available() else "cpu"


# create the MLP (Multi Layer Perceptor)
class OurMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # These are not in a sequential block, so we can connect them as we please
        # This allows us to feed forward and back as we please
        # This is what a convolutional NN is doing
        self.input_layer = nn.Linear(28 * 28, 50)
        self.hidden1 = nn.Linear(50, 100)
        self.hidden2 = nn.Linear(100, 50)
        self.output_layer = nn.linear(50, 10)
        self.activation = nn.Sigmoid()
        self.flatten = nn.Flatten()
        # The flatten layer takes a muti dim input and provides as output a mono-dim tensor

    def forward(self, x):
        # We are redefining the same sequential steps from the previous model, but manually
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        x = self.flatten(x)  # flatten the data for the mlp (mlp just takes 1D vector)
        logits = self.mlp(x)
        return logits


# TO actually make and train a model we first need to initialize:
model = OurMLP().to(device)
# This gives the model to our best hardware (ie, if poss the GPU)
# We have to provide the hyper parameters, these are the things outside of the model

epochs = 2  # How many times the model should iterate over whole dataset

batch_size = 16  # How many samples from dataset should be taken at once
# In this case 16 images are taken for each loop. Also could be called step size
# It passes through the 16 images, then computes the weights (the gradient descent)
# and backpropogates the results only at the end of each batch

learning_rate = 0.001  # THe learning rate is how fast backpropogation has it's effects
# A large learning rate may cause the model to "jump" over an optimal solution
# You may get stuck in a local minimum
# Too low will cause training to be slow

# Learning rate can be adjusted as the model runs, to reduce to improve the accuracy of the parameter
# adjustment to result in a better model


# DEFINE THE LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()
# This is the "soft max ..." function, it turns the logits into a probability table

# DEFINE THE OPTIMIZER
# this function that computes the derivatives, so it is what is used to backpropogate to improve the model
# It's telling the model which mathematical functions to use
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# SGD is the original optimizer, however AdamW gets much better results!!
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# CREATE THE DATALOADER
# The dataloader gets the data from disc and gives it to the model
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# define accuracy metric
metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)
# This is the metric we use to define how good our model is
# If both computes the accuracy, but also stores the values as the model
# gets trained

# We're just trying to define classes, as labelled, to it's multiclass

# Defining our training loop


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # X and y are the batch size set when defining the dataloader (in this case 16)

        # Because the model is on the device (GPU?) we need to move the data there too
        X = X.to(device)
        y = y.to(device)

        # compute the prediction and the loss

        # So each loop we put a bunch of inputs into the model, the compare them with
        # the correct answers. We feed that info back into the model using the optimizer
        # which will train the model and update the weights

        pred = model(X)  # predictions the model makes
        loss = loss_fn(
            pred, y
        )  # the comparison between predictions and correct answers

        # These lines MUST be called in this order
        loss.backward()  # back propogation
        optimizer.step()  # the optimizations
        optimizer.zero_grad()  # Re-zero the gradients before/after every optimization step

        if batch % 20 == 0:
            loss_val, current_batch = loss.item(), (batch + 1) * len(X)
            print(f"Current loss: {loss_val}, completion: [{current_batch}/{size}]")
            accuracy = metric(pred, y)
            print(f"Current accuracy: {accuracy}")
    acc = metric.compute()
    print(f"----------------\nFinal accuracy: {acc}\n----------------")
    metric.reset()  # Now we can use the metric again for testing/validation


# The testing loop is about the same as training, except we don't need to update the model
def test_loop(dataloader, model):
    size = len(dataloader)

    # disable weight update!
    with torch.no_grad():
        for X, y in dataloader:
            # move data to correct device:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            acc = metric(pred, y)
    # compute final accuracy
    acc = metric.compute()
    print(f"\nTesting accuracy: {acc}\n----------------")
    metric.reset()


for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model)

print("LESSSSSSSSSSSSSSGOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
