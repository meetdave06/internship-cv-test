# random-cv-tasks

## Libraries required

* numpy
* pytorch
* cv2
* dlib

The models used in this repository can be found in the models directory

## Task 1

In this task, the background of the person in the video is replaced. we use a segmentation model to detect and mask the person from image. After that
we use alpha-blending to merge the person mask with the background.

Video :-  https://drive.google.com/open?id=1TFyWy18KwOST3r9EzRbK8Vo92ilRZoMK


## Task 2

We use dlib face detection model to detect the face of the person in image. After that, we crop the face and use the dlib facial
landmark detector model to get the 68 facial landmarks. Once we have the landmarks, we can use that to place the hat on the person's head

Video :- https://drive.google.com/open?id=1ASO253OOeaM53caLkvvVyNn5b8Pc4YgA

## Task 3

In this task, we use OpenCV mouse callback functions to write/draw on image. In a webcam/video stream, as the image keeps on changing, we maintain
a transparent frame to draw on which is then overlayed on the video using alpha-blending creating a effect of writing/drawing on video in real-time.

Video :- https://drive.google.com/open?id=19SKNo47gR6V0KtUCroaLL2AlGGbdKPDu


# Run the code

In order to try out the above tasks, cd into the respective task folder and run
`python task1.py`. The effects can be seen real time and the video will be saved as well in the images folder.

* Developer :- Meet Dave
* E-mail :- meetkirankum@umass.edu
