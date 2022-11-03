import cv2
import matplotlib.pyplot as plt
import numpy as np

# adding extra pading to make it squre
# making image squre is important because we have to resize image to 28*28
def add_padding(img):

    img = np.pad(img,pad_width=50)
    (a,b)= np.shape(img)
    top,down,left,right =0,0,0,0

    if a>b:
        pad = (a-b)//2
        left = pad
        right =pad
    else:
        pad = (b-a)//2
        top = pad
        down = pad
    
    padding=((top,down),(left,right))
    img = np.pad(img,padding,mode='constant',constant_values=0)
    return img

# croping white space around the image 
def crop_image(img):
    x1,x2,y1,y2 = 0,0,0,0
    (thresh, img) = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    # plt.imshow(img)
    # plt.show()
    for i in range(len(img)):
        # print(str(sum(1 for e in img[i] if e == 255)) + ' , '+ str(len(img[i])))
        if sum(1 for e in img[i] if e == 255) != len(img[i]):
            if x1 == 0:
                x1 = i
            x2 = i
    # print(str(x1)+','+str(x2))        
    for i in range(len(img.T)):
        if sum(1 for e in img[:,i] if e == 255)!= len(img[:,i]) :
            if y1 == 0:
                y1 = i
            y2 = i
    return img[x1:x2,y1:y2]

# spliting multi digit image to seprate digits
def get_digits(imageName):
    #reading image from file
    image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)   
    plt.imshow(image)
    # plt.show()

    #removing white space from around of image
    cropedImage = crop_image(image)
    plt.imshow(cropedImage)
    # plt.show()

    x1,x2,y1,y2 = 0,len(cropedImage)-1,-1,0
    acceptableImageLen = 2
    col = 0
    images = []
    c = 0;
    while c < len(cropedImage.T) :
        # finding starting anding column of each image
        for i in range(col, len(cropedImage.T)):
            c = c+1
            # print(sum(1 for e in cropedImage[:,i] if e == 0))
            if sum(1 for e in cropedImage[:,i] if e == 0) >= acceptableImageLen and y1 == -1:
                y1 = i
            if sum(1 for e in cropedImage[:,i] if e == 255) >= len(cropedImage[:,i])-acceptableImageLen and y1 != -1:
                y2 = i-1
                col = i+1
                break
        # print('x1:'+str(x1)+' x2:'+str(x2)+' y1:'+str(y1)+' y2:'+str(y2)+' c:'+str(c))
        if(c == len(cropedImage.T)):
            y2 = c-1
        if y2 - y1 > 1:
            result = cropedImage[x1:x2, y1:y2]
            result = crop_image(result)
            result = np.invert(result)
            result = add_padding(result)
            images.append(result)
        y1,y2 = -1,0
    return images

##############################################################################

# images = get_digits('webcam/webcammkhsgqtdzv.png')
# # print('------------------------------------------')
# for i in range(len(images)):
#     plt.imshow(images[i])
#     plt.show()
    # img = images[i]
    # img = crop_image(img)
    # img = add_padding(images[i])
    # plt.imshow(img)
    # plt.show()