import cv2
import numpy as np
import torch
import torchvision
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

###################### properties ##########################################
n_epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
# momentum is designed to accelerate learning
momentum = 0.5

# When an image is transformed into a PyTorch tensor, the pixel values are scaled between 0.0 and 1.0. 
# In PyTorch, this transformation can be done using torchvision.transforms.ToTensor().

# Normalization helps get data within a range and reduces the skewness which helps learn faster and better.
# Normalization in PyTorch is done using torchvision.transforms.Normalize()

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

####################### NET CALSS #####################################################
class Net(nn.Module):

# 1 input (image), outputting 10 features
# 10 input (image), outputting 20 features
# Dropout is a regularization technique that prevents neural networks from overfitting

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

    # In the case of a Convolutional Neural Network, the output of the convolution will be passed through the activation function. (ReLU)
    # dropout remove half of neurons
    # we use the softmax activation function for our last layer model
    def forward(self, x):
        # input size is [64, 1, 28, 28]
        mp2 = F.max_pool2d(self.conv1(x), kernel_size=2)
        # print(f"maxpool 2: {mp2.size()}")
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        # print(f"ReLU of conv1: {x.size()}")
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(f"ReLU of conv2: {x.size()}")
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
###################### INIT ############################
network = Net()
# optimizer implements a step() method, that updates the parameters. it will be used in train method
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

continued_network = Net()
continued_optimizer = optim.SGD(continued_network.parameters(), lr=learning_rate,momentum=momentum)

modelPath = '/model.pth'
optimizerPath = '/optimizer.pth'

###################### DO Train ###########################

def do_train():
    for epoch in range(1, n_epochs + 1):
        train(epoch)

####################### TRAIN #####################################################
def train(epoch):
  #sets the mode to train
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    # for every mini-batch during the training phase, we typically want to explicitly set the gradients to zero before starting to do backpropragation 
    # because PyTorch accumulates the gradients on subsequent backward passes. 
    optimizer.zero_grad()

    #set input and get output
    output = network(data)

    #calc loss
    loss = F.nll_loss(output, target)
    loss.backward()

    # update parameters
    optimizer.step()

    # A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
    torch.save(network.state_dict(), modelPath)
    torch.save(optimizer.state_dict(), optimizerPath)

########################## LOAD TRAINED NETWORK 
def load_trained_net():
    network_state_dict = torch.load(modelPath)
    continued_network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load(optimizerPath)
    continued_optimizer.load_state_dict(optimizer_state_dict)

########################## TEST OWN IMAGES ####################
def predictSingleImage(image, orginalValue):
  continued_network.eval()
  # The requires_grad argument tells PyTorch that we want to be able to calculate the gradients for those values. 
  # However, the with torch.no_grad() tells PyTorch to not calculate the gradients
  with torch.no_grad():
    single_loaded_img = image #test_loader.dataset.data[0]
    single_loaded_img = single_loaded_img[None, None]
    single_loaded_img = torch.from_numpy(single_loaded_img)
    single_loaded_img = single_loaded_img.type('torch.FloatTensor') # instead of DoubleTensor


    out_predict = continued_network(single_loaded_img)
    # print(f"output : {out_predict}")
    pred = out_predict.max(1, keepdim=True)[1]
    return pred.item()
###############################################
def predict_digits(images):
    predict = []
    for i in range(len(images)):
        img = images[i]
        img = cv2.resize(img, (28, 28))
        predict.append(predictSingleImage(img, i))
        # plt.imshow(img)
        # plt.show()
        # singleTest(img2, i)
        # plt.imshow(img2)
        # plt.show()
        # print('----------------')
    return predict

# do_train()