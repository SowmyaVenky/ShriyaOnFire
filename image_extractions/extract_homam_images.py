import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import time
import sys
import uuid
  
try: 
      
    # creating a folder named data 
    if not os.path.exists('data'): 
        os.makedirs('data') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 
  
# frame 
currentframe = 0

print( 'Arguments passed length: ' + str(len(sys.argv)))

if( len (sys.argv) != 2 ):
    print( 'Please send the video to process as an argument...')
    exit()

print ('argument list', sys.argv)
video_name = sys.argv[1]
print ("Processing video {}...".format(video_name))
prefix = str(uuid.uuid4())

# Read the video from specified path 
cam = cv2.VideoCapture(video_name) 

cam.set(3, 640)
cam.set(4, 480)

while(True): 
      
    # reading from frame 
    ret,frame = cam.read() 
  
    if ret: 
        # if video is still left continue creating images 
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
              
        if currentframe % 50 == 0: 
          name = './data/images/' + prefix + '_frame' + str(currentframe) + '.jpg'      
          print("Writing frame : " + str(currentframe))  
          retval = cv2.imwrite(name, frame)
          if retval:
            print("Image saved...") 
    else: 
        break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 