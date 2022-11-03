# Multi-digit Handwritten number Recognition in PyTorch

The purpose of this article is to build a simple convolutional neural network in PyTorch and train it to recognize handwritten digits using the MNIST dataset. Minst dataset contains `70,000` samples in `28*28` pixel handwritten images which we use them to train our model. finally we can test our model by mnist dataset as well as multi-digit handwritten which we take from the webcam.

## loading and training the model

it is simple. first of all, we have to load mnist dataset in two categories of **train** and **test** data sets.
for this, we use `torch.utils.data.DataLoader` to download and load. (if it was downloaded already, it does not download again).

```
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose(...),
							...)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose(...),
							...)

```

The place that I mention the images in mnist data set is `28*28` grayscales. it means all images are in black and **white color with black background**. so, next, we will see that we need to do some preprocessing tasks on our test images to convert them to `28*28`, `black&whilte`, and `black background` images before testing them with our model.

## Building the Network.
we start building our model by creating a `Net` class. we use two 2-D convolutional layers and two linear layers. Also, we choose rectified linear units for our activation function.

```
class Net(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ...
```
### forward function
This will represent our feed-forward algorithm. it returns outputs as a prediction for given input x.
> Note: we don't call forward function for getting output, we call the whole model to perform a forward pass and output prediction

## train model
after building our model now we can train it. first, we set the mode of the network to training mode and then we iterate over training data once per each epoch.
another point worth mentioning is that Neural network modules as well as optimizers have the ability to save and load their internal state using `.state_dict()`.With this we can continue training from previously saved state dicts if needed - we'd just need to call `.load_state_dict(state_dict)`. 

```
def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    ...
    torch.save(network.state_dict(), modelPath)
    torch.save(optimizer.state_dict(), optimizerPath)
```

so far we could train our network and save it.
now we want to load the trained network and test our own handwritten multi-digit numbers.

```
def load_trained_net():
    network_state_dict = torch.load(modelPath)
    continued_network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load(optimizerPath)
    continued_optimizer.load_state_dict(optimizer_state_dict)
```

## Capturing image from webcam
we could simply capture our image from a web cam by some line of codes.

```
def get_image_from_webcam():
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 32:
            # SPACE pressed
            img_name = "webcam/webcam{}.png".format(rand)
            cv2.imwrite(img_name, frame)
            break
    return imageName
```
then we pass it to our `digits.py`.


## Separating digits
next, we need to separate the digits and pass them separately to our model and get the predicted number.

> Because these pieces of code are not so intelligent we should be careful and follow some rules for taking images.
> for example, the pen of our drawing for digits should be in big size and a little bit thick. Also, we have to take only the inside of the paper and the background should only be the white paper.

separating digits from the image and resizing it to fit our model is the most challenging part of this task.
we should **recognize digits**, **converting it to black&whilte**, **changing background to black** and set some **padding** to make it **squre** are tasks that we should done them.

in `digits.py` we have some functions that do these tasks for us.

```
def get_digits(imageName):
    #reading image from file
    image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)   
    
    #removing white space from around of image
    cropedImage = crop_image(image)
    
    while c < len(cropedImage.T) :
        for i in range(col, len(cropedImage.T)):
            ...
			finding starting anding column of each image
			...
        #geting each digit
		result = cropedImage[x1:x2, y1:y2]
		
		#croptin white space again
		result = crop_image(result)
		
		#changing background to black
		result = np.invert(result)
		
		#add padding and making it squre
		result = add_padding(result)
		
		images.append(result)
        
    return images
```
first, we pass the image location to `get_digits`, inside this function we remove all white space from around the image.
then by recognizing black and white pixels we can recognize each digit. we have to add padding and make it square because we have to resize them to 28*28, so we don't want to re-shape the image!

finally, by passing each digit to our model we can recognize the number of multi-digit handwritten numbers.

```
def predictSingleImage(image, orginalValue):
  continued_network.eval()
  with torch.no_grad(): 
    single_loaded_img = image
    ...
    out_predict = continued_network(single_loaded_img)
    pred = out_predict.max(1, keepdim=True)[1]
    return pred.item()
```

