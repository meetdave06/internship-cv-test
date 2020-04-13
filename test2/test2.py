__author__ = "Meet Dave"
__version__ = "1.0"
__maintainer__ = "Meet Dave"
__email__ = "meetkirankum@umass.edu"


import matplotlib.pyplot as plt
import numpy as np
import dlib
import cv2
from utils import shape_to_np, calculate_face_angle, rotate_bound, make_video

prototxt_path = "../models/deploy.prototxt"
caffemodel_path = "../models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
shape_predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
face_detector_path = "../models/mmod_human_face_detector.dat"
filter_path = "../images/nikecap.png"
video_path = "../images/test2.avi"


# Load dlib model
dlib_face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
# Load shape predictor
facial_landmark_detector = dlib.shape_predictor(shape_predictor_path)

# Load filter image
filter = cv2.imread(filter_path,-1)
filter = cv2.cvtColor(filter, cv2.COLOR_BGRA2RGBA)

cap = cv2.VideoCapture(0)
ret, img = cap.read()
height = img.shape[0]
width = img.shape[1]

video_download = make_video(video_path,width,height)

resize_shape = 300


while(True):

    ret, img = cap.read()

    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, (resize_shape, resize_shape))

        # Detect face
        faceRects = dlib_face_detector(img_resize, 1)

        if len(faceRects)==0:
            continue
    
        # Get top left, bottom right coords
        box_dlib = faceRects[0].rect.left(),faceRects[0].rect.top(), faceRects[0].rect.right(), faceRects[0].rect.bottom()
        # Resize to match original image size
        # box_dlib = box_dlib *  np.array([width/resize_shape,height/resize_shape,width/resize_shape,height/resize_shape])
        # # Get face width
        # face_width = np.int(faceRects[0].rect.width() * width/resize_shape)
        # # Get face height
        # face_height = np.int(faceRects[0].rect.height() * height/resize_shape)
        # # Convert to int to draw rect and pass to shape predictior
        x1_d,y1_d,x2_d,y2_d = box_dlib
        # dlib_rect = dlib.rectangle(x1_d,y1_d,x2_d,y2_d)

        # Get facial landmarks
        shape_dlib = facial_landmark_detector(img_resize,faceRects[0].rect)
        shape_dlib = shape_to_np(shape_dlib)

        # face width
        forehead_width = np.int(faceRects[0].rect.width())
        # forhead_height
        forehead_height = np.int(forehead_width * 4/5)

        # Calculate face angle
        leftmost_eyebrow_point = shape_dlib[17]
        rightmost_eyebrow_point = shape_dlib[26]
        face_angle = calculate_face_angle(leftmost_eyebrow_point,rightmost_eyebrow_point)

        # Rotate filter according to face angle
        foreground_rot = rotate_bound(filter,face_angle)

        # Resize filter to forehead width
        resized_filter = cv2.resize(foreground_rot,(forehead_width,forehead_height))   

        # where to place

        roi = img_resize[y1_d-forehead_height:y1_d,x1_d:x1_d+forehead_width]

        alpha_mask = resized_filter[:,:,3]/255

        background = roi * (1-alpha_mask[:,:,None])

        foreground = resized_filter[:,:,0:3] * alpha_mask[:,:,None]

        final_add_f = cv2.add(background,foreground) 

        img_resize[y1_d-forehead_height:y1_d,x1_d:x1_d+forehead_width,:] = final_add_f

        # Resize to original

        img = cv2.resize(img_resize,(width,height))
        img = img[...,::-1]
        final = (img).astype(np.uint8)
        cv2.imshow('frame',final)
        video_download.write(final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindow