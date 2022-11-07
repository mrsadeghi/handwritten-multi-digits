import os
from webcam import get_image_from_webcam
from digits import get_digits
from mnistNet import predict_digits,load_trained_net, do_train


# do_train()
load_trained_net()
# imageName = get_image_from_webcam()
imageName = 'webcam/webcammkhsgqtdzv.png'
images = get_digits(imageName)
val = predict_digits(images)
clear = lambda: os.system('cls')
clear()
val_str = ''.join(map(str, val))
print('Predicted value is : '+ str(''.join(val_str)))
