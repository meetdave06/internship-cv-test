__author__ = "Meet Dave"
__version__ = "1.0"
__maintainer__ = "Meet Dave"
__email__ = "meetkirankum@umass.edu"



import cv2
import numpy as np
from make_video import make_video

drawing = False # true if mouse is pressed
ix,iy = -1,-1
video_path = "../images/test3.avi"


# mouse callback function
def draw_onscreen(event,x,y,flags,param):
    global ix,iy,drawing,transp,alpha,height,width

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:

            cv2.line(transp,(ix,iy),(x,y),(0,0,255),3)
            ix = x
            iy = y


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(transp,(ix,iy),(x,y),(0,0,255),3)
            
    elif event == cv2.EVENT_MBUTTONDBLCLK:
        transp = np.zeros((height,width,3))
        alpha = np.zeros((height,width,3))


            

cap = cv2.VideoCapture(-1)
ret, img = cap.read()

height = img.shape[0]
width = img.shape[1]
transp = np.zeros((height,width,3))
alpha = np.zeros((height,width,3))
video_download = make_video(video_path,width,height)


while(ret):
    cv2.namedWindow('test3')
    cv2.setMouseCallback('test3',draw_onscreen)
    ret, img = cap.read()
    if ret:
        alpha[transp>0] = 1
        final = img * (1-alpha) + transp*alpha
        cv2.imshow('test3',(final).astype(np.uint8))
        video_download.write((final).astype(np.uint8))
        if cv2.waitKey(20) & 0xFF == 27:
            break
cv2.destroyAllWindows()