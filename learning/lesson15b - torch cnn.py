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
class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=5, kernel_size=3
            ),  # 1 input channel because greyscale
            # we can choose as many outputs as we want, more channels means more data can be extracted
            # output channels = 5 means 5 filters are being applied
            # kernel size of 3
            nn.ReLU(),  # activation function
            # Could be called "efficient sigmoid"
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3),
            nn.ReLU(),
        )  ############# We've slightly reduced dimensions but increased channels 10-fold, doesn't this increase training time again???
        ############# For "horizontal" images do rectangular kernels/strides work better?
        self.mlp = nn.Sequential(
            nn.Linear(in_features=24 * 24 * 10, out_features=10),
            # The in features here need to be the same as the output from the conv layer
            nn.Dropout(0.5),
            # THis will randomly remove 50% of the connections in a batch
            # THis stops the edges from being modified in a specific batch run
            # This can avoid overfitting
            # THESE MUST BE REMOVED WHEN USING THE MODEL!!!!
            # using the function model.eval() will set them to 0
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
        )

    def forward(self, x):
        # We are redefining the same sequential steps from the previous model, but manually
        x = self.cnn(x)
        x = torch.flatten(x, 1)  # (input data, dimension to flatten on)
        x = self.mlp(x)


############ FROM HERE IS IDENTICAL TO THE ANN DONE IN PREVIOUS LECTURES ############

# TO actually make and train a model we first need to initialize:
model = OurCNN().to(device)
# This gives the model to our best hardware (ie, if poss the GPU)
# We have to provide the hyper parameters, these are the things outside of the model

# THis is just during development to figure out the size of the vector being passed into the MLP
# This tells use the shape of the model that is coming out of the convolution layer
test_x = torch.rand(
    (1, 28, 28)
)  # But the actual input will be batch size * channels * height * width
test_y = model(test_x)
exit()

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
