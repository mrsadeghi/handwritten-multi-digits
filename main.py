from webcam import get_image_from_webcam
from digits import get_digits
from mnistNet import predict_digits,load_trained_net, do_train


# do_train()
load_trained_net()
imageName = get_image_from_webcam()
imageName = 'webcam/webcammkhsgqtdzv.png'
images = get_digits(imageName)
val1 = predict_digits(images)
print(*val1)


# images = get_digits('webcam/webcammkhsgqtdzv.png')
# val2 = predict_digits(images)
# print(*val2)
