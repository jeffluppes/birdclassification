######## Bird object detection using SSD and a classification model #########
#
# Author: Jeffrey Luppes (https://github.com/jeffluppes) 
# Date: 2020-08-10
# Description: 
# This script using the Mobilenet-v1 SSD to perform object detection. If a bird is detected 
# the script continues to figure out what kind of bird it is by running a classifer on the 
# extracted ROI from the pi camera stream. Both models are tensorflow lite models, but only
# the bird classification model is unquantized. 
#
# This script was heavily based on existing work by the following people:
#
# Evan Juras, who made a pi camera object detector (https://github.com/EdjeElectronics)
# Leigh Johnson who made a pan-tilt object tracker (https://github.com/leigh-johnson/rpi-deep-pantilt)
# as well as various others in the community.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
 
def center_image(img):
    '''Convenience function to return a centered image'''
    size = [256,256]
    
    img_size = img.shape[:2]
    
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	    # Return the most recent frame
        return self.frame

    def stop(self):
	    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.1)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

BM_MODEL = 'model.tflite' #todo make quant version of this

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
PATH_TO_BM = os.path.join(CWD_PATH,BM_MODEL)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# labels for the bird names consist out of an a-z list of bird species used for training
birdclassnames = pickle.load(open('birdclassnames.p', 'rb'))

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(f'Loaded object detection model from {PATH_TO_CKPT}')
    birdmodel = Interpreter(model_path=PATH_TO_BM,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(f'Loaded bird model from {PATH_TO_BM}')
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    print(f'Loaded object detection model from {PATH_TO_CKPT}')
    birdmodel = Interpreter(model_path=PATH_TO_BM)
    print(f'Loaded bird model from {PATH_TO_BM}')

interpreter.allocate_tensors()
birdmodel.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# same stuff for bird model
# Get input and output tensors.
bm_input_details = birdmodel.get_input_details()
bm_output_details = birdmodel.get_output_details()
bm_input_shape = bm_input_details[0]['shape']
bm_type = bm_input_details[0]['dtype']
print(f'Bird model is using the {bm_type} datatype')

# normalize if using unquantized mobilenet
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()

    # flip the frame because the camera is upside down in my appartment
    frame = cv2.flip(frame, -1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))]
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results 
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((labels[int(classes[i])] == 'bird') and (scores[i] <= 1.0) and (scores[i] > min_conf_threshold)):

            # start timer
            predtime = time.time()

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            # Also added +5 pixels to aid in prediction for slightly bigger box
            ymin = int(max(1,(boxes[i][0] * imH)-5))
            xmin = int(max(1,(boxes[i][1] * imW)-5))
            ymax = int(min(imH,(boxes[i][2] * imH)+5))
            xmax = int(min(imW,(boxes[i][3] * imW)+5))

            if(ymax-ymin > 0 and xmax-xmin > 0): # for some reason we sometimes get a prediction of shape (x, 0), this fixes that
                
                # make a sub image
                roi = frame_rgb[ymin:ymax, xmin:xmax]

                # resize the matrix
                if(roi.shape[0] > roi.shape[1]):
                    tile_size = (int(roi.shape[1]*256/roi.shape[0]),256)
                else:
                    tile_size = (256, int(roi.shape[0]*256/roi.shape[1]))

                #centering
                x = center_image(cv2.resize(roi, dsize=tile_size))

                #output should be 224*224px
                x = x[16:240, 16:240]

                # check if we're using a quant model or not. If we are, convert the input data
                if bm_type == np.float32:
                    # rescale by casting to float and dividing by max
                    x = x.astype(np.float32)
                    x = x / 255

                x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 224, 224, 3) and should be a float

                #get predictions
                birdmodel.set_tensor(bm_input_details[0]['index'], x)
                birdmodel.invoke()
                preds = birdmodel.get_tensor(bm_output_details[0]['index'])
                species = birdclassnames[preds.argmax(axis=-1)[0]]

                # print a sanity check
                print(f'Found a {species} at the feeding station!')
                filename = str(predtime).replace('.', '')

                # store roi for future training data purposes
                cv2.imwrite(f'rois/{filename}_roi_{species}.png', frame)
		
                # Draw a rectange
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 144, 255), 2)

                # Draw label
                score = preds[0][np.argmax(preds)]
                label = '%s: %d%%' % (species, int(score*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (0, 144, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2) # Draw label text
                cv2.imwrite(f'detections/{filename}_detection_{species}.png', frame)
                
                print(f'Prediction time: {time.time()-predtime} seconds')

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()