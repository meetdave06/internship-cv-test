__author__ = "Meet Dave"
__version__ = "1.0"
__maintainer__ = "Meet Dave"
__email__ = "meetkirankum@umass.edu"

# Load libraries

import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from torchvision import models
from torchvision import transforms
from make_video import make_video




# Load pretrained model
deeplapv3_101 = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Load background image

background_path = "../images/books-seats.png"
background = cv2.imread(background_path)
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

video_path = "../images/test1.avi"


# Webcam stream

cap = cv2.VideoCapture(0)

ret, img = cap.read()
height = img.shape[0]
width = img.shape[1]

video_download = make_video(video_path,width,height)


background = cv2.resize(background, (width,height))
background = background.astype(float)

# Preprocess class

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


while(True):

    ret, img = cap.read()

    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess image
        input_img = preprocess(img)
        # Creating a batch dimension
        input_batch = input_img.unsqueeze(0)

        # Inference
        output = deeplapv3_101(input_batch)['out'][0]
        final_output = output.argmax(dim=0)

        # Just keep person class and make everything else background
        person_output = torch.zeros_like(final_output)
        person_output[final_output == 15] = 1

        img_resize = cv2.resize(img,(256,256))
        # Get person segmentation
        foreground = img_resize * person_output.numpy()[:,:,None]
        foreground = foreground.astype(float)
        foreground_orig_size = cv2.resize(foreground,(width,height))


        # Create alpha mask for blending
        th, alpha = cv2.threshold(foreground_orig_size,0,255, cv2.THRESH_BINARY)
        # Smooth the edges for smooth blending
        alpha = (cv2.GaussianBlur(alpha, (7,7),0))/255

        final = foreground_orig_size * alpha + background * (1 - alpha)
        final = final[...,::-1]
        final = (final).astype(np.uint8)
        cv2.imshow('frame',final)
        video_download.write(final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()