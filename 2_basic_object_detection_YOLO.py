import cv2 as cv2
import cv2 as cv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import warnings
warnings.filterwarnings("ignore")

# read in cleaned csv
fr = pd.read_csv("FR_cleaned_data.csv")

# YOLO/object detection setup
modelConfiguration = "yolov3-spp.cfg"
modelWeights = "yolov3-spp.weights"
whT = 320
confThreshold = 0.9
nmsThreshold = 0.5

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# function to determine if cow is in frame with pre-set probabilities
    # if cow found with a given probability, draw the bounding box
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT) - w/2), int((det[0]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))


    # approach 1
    #indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # approach 2
    ## need to ensure we are choosing the cow bounding box with the largest confidence ****

    if len(confs) > 0:

        i = confs.index(max(confs))

        #if len(indices) > 0:
            #for i in indices:
        if classIds[i] == 19:

                    box = bbox[i]
                    x,y,w,h = box[0], box[1], box[2], box[3]

                    ### FOR TESTING ###
                    #cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
                    #cv2.putText(img, f'COW - {int(confs[i]*100)}%', (x, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 3)

                    return True, x, y, w, h

        else:

            return False, False, False, False, False

    else:
        return False, False, False, False, False

    return False, False, False, False, False

## function to rescale image
def rescaleFrame(frame, h=240, w=427):
    dimensions = (int(w), int(h))

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# initialise lists
d_img_index = []
d_lot_index = []
d_weight = []

# initialise clip index counter
img_index = 0

## loop through lots/clips
for i in range(len(fr)):

### For Testing ###
#for i in range(1):

        try:

            # choose single clip
            capture = cv2.VideoCapture("FR_clips/lot" + str(i) + ".mp4")

            # initialise counter as a stopping criteria
            counter = 0
            stoping_criteria = 20

            # loop through each frame in the single clip
            while True:

                isTrue, frame = capture.read()

                # object detection setup - find objects in single frame
                blob = cv2.dnn.blobFromImage(frame, 1/255, (whT,whT), [0,0,0], 1, crop = False)
                net.setInput(blob)
                layerNames = list(net.getLayerNames())
                outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(outputNames)

                # determine if cow exists with a certain probability
                cond, x, y, w, h = findObjects(outputs, frame)

                if cond == True:

                    ### FOR TESTING ###
                    # show the frame if cow found
                    ###cv2.imshow('Video', frame)
                    # draw a bounding box
                    ###findObjects(outputs, frame)


                    # reset counter
                    counter = 0
                    # save relevant images and appropriate weights
                    d_weight.append(fr.cat_weightKgs[i])
                    d_img_index.append(img_index)
                    d_lot_index.append(i)

                    # first attempt#
                    # cv2.imwrite("FR_images" + "\\img_" + str(img_index) + ".jpg", rescaleFrame(frame))
                    # img_index += 1

                    # save the rescaled and cropped image**
                    # crop based on the bounding box, need to add clearence - set pixles each side
                    ## TO DO - play around with this clearance**

                    ### For Testing ###
                    #cv2.imshow("video", frame)

                    # greyscale
                    #cv2.cvtColor(rescaleFrame(cropped), cv2.COLOR_BGR2GRAY)

                    cropped = frame[max(y-250,0):min(y+h+250, 720), max(x-250,0):min(x+w+250, 1280)]
                    cv2.imwrite("FR_images_3" + "\\lot_" + str(i) + "_img_" + str(img_index) +".jpg" , rescaleFrame(cropped))

                    img_index += 1

                else:

                    counter += 1

                # is stopping criteria met
                if counter > stoping_criteria:
                    break

                if cv2.waitKey(20) & 0xFF == ord('d'):
                    break

            capture.release()
            cv2.destroyAllWindows()


        except:

            capture.release()
            cv2.destroyAllWindows()


# write corresponding weights and clip index to dataframe
fr_images = pd.DataFrame(d_lot_index, columns = ["lot_index"])
fr_images["img_index"] = d_img_index
fr_images["weight"] = d_weight
fr_images.to_csv("fr_images.csv")

print("Successfully created CNN input data")


# This is a naive implementation, will need to be improved


## possible improvements

# 1. use the bounding box to crop and then re-size for input to CNN
    # will need to expand box to account for that underprediction
    # find the man aswell and black this out?? maybe .. Q. How to deal with this noise
# 2. Move away from YOLO, learn how to create your own object detection that only detects side and front views
    # we may have a few objects, side profile, front, back etc..
# 3. Advanced object detection ~ mask RCNN and advanced CV prediction
# 4. Ideally, we can programatically remove lots with > 1 cattle**

