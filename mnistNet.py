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
momentum = 0.5
log_interval = 10

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
    def __init__(self):
        super(Net, self).__init__()
        # 1 input (image), outputting 10 features
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 10 input (image), outputting 20 features
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout is a regularization technique that prevents neural networks from overfitting
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
###################### INIT ############################
network = Net()
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
    optimizer.zero_grad()
    #set input and get output
    output = network(data)
    #calc loss
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
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
  with torch.no_grad():
    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
   
    single_loaded_img = image#test_loader.dataset.data[0]
    single_loaded_img = single_loaded_img[None, None]
    single_loaded_img = torch.from_numpy(single_loaded_img)
    single_loaded_img = single_loaded_img.type('torch.FloatTensor') # instead of DoubleTensor


    out_predict = continued_network(single_loaded_img)
    # print(out_predict)
    pred = out_predict.max(1, keepdim=True)[1]
    return pred.item()
    # print(str(pred.item()))
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

