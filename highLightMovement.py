#Highlight Movement
import cv2
import numpy as np
from skimage.measure import compare_ssim
import imutils
import time
import sys

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

videoFile           = "firstBasement.mp4"
lastFrame           = None
boundingAreaSize    = 20
window_name         = "Video"
scale_percent       = 75 # percent of original size
frameIndex          = 0

#cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


#Open Video
cap         = cv2.VideoCapture(videoFile)
videoLength    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def on_change(value):
    global frameIndex
    frameIndex = value
    print("value = ", value)
    cap.set(cv2.CAP_PROP_POS_FRAMES, value)
    
    


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

if cap.isOpened():
    # Capture frame-by-frame
    ret, lastFrame = cap.read()
    
    width = int(lastFrame.shape[1] * scale_percent / 100)
    height = int(lastFrame.shape[0] * scale_percent / 100)
    dim = (width, height)
    lastFrame = cv2.resize(lastFrame.copy(), dim, interpolation = cv2.INTER_AREA)
    frameIndex += 1

cv2.createTrackbar('slider', window_name, frameIndex, videoLength, on_change)


# Read until video is completed
while(cap.isOpened()):
    ret, frame = cap.read()
    frameIndex += 1
    if ret == True:  
        start = time.time()
        #width = int(frame.shape[1] * scale_percent / 100)
        #height = int(frame.shape[0] * scale_percent / 100)
        #dim = (width, height)
        
        # resize image
        frame = cv2.resize(frame.copy(), dim, interpolation = cv2.INTER_AREA)
        print("Time 1 ", time.time()-start)

        start = time.time()
        # convert the images to grayscale
        grayA = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(lastFrame, cv2.COLOR_BGR2GRAY)
        print("Time 2 ", time.time()-start)

        start = time.time()
        #(score, diff) = compare_ssim(frame, lastFrame, full=True, multichannel=True)
        (score, diff) = compare_ssim(grayA, grayB, full=True, multichannel=False)
        diff = (diff * 255).astype("uint8")
        #print("SSIM: {}".format(score))
        lastFrame = frame.copy()
        print("Time 3 ", time.time()-start)

        start = time.time()
        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
        print("Time 4 ", time.time()-start)

        # loop over the contours
        start = time.time()
        for c in cntsSorted[0:5]:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            area = cv2.contourArea(c)
            if area > boundingAreaSize:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.drawContours(frame, c, -1, (0, 255, 0), 2)
                #cv2.fillPoly(frame, pts =[c], color=(0,255,0))
                #time.sleep(1.5)
                #break

            # Display the resulting frame
            #frameIndex= int(cv2.getTrackbarPos('slider',window_name))
            cv2.putText(frame, "Frame : " + str(frameIndex), (20, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow(window_name,frame)
            
        print("Time 5 ", time.time()-start)
        cv2.setTrackbarPos('slider', window_name, frameIndex)
        
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        #break

    # Break the loop
    else: 
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
