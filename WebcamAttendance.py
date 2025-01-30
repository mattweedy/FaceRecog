import os
import cv2
import numpy as np
import face_recognition
from threading import Thread
from datetime import datetime
from VideoGrab import VideoGrab
from VideoShow import VideoShow
from FrameProcessor import FrameProcessor

# implementing multithreading from : https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/

# defining vars
path = 'ImagesWebcam'
images = []
classNames = []
files = os.listdir(path)

# loop through files
# append images and classNames
for fileName in files:
    currImg = cv2.imread(f'{path}/{fileName}')
    images.append(currImg)
    classNames.append(os.path.splitext(fileName)[0].replace('_', ' '))
    
def compareFaces(encodeCurrFrame, facesCurrFrame, frame):
    for encodeFace, faceLoc in zip(encodeCurrFrame, facesCurrFrame):
        # ensure encodeFace is a numpy arr
        if isinstance(encodeFace, list):
            encodeFace = np.array(encodeFace)
        
        # compare current encodeFace to all known faces
        matches = face_recognition.compare_faces(encodeListKnownFaces, encodeFace)
        # find dist
        faceDis = face_recognition.face_distance(encodeListKnownFaces, encodeFace)
        
        # set index of best match
        matchIndex = np.argmin(faceDis)
        
        # print the name of the match if it exists
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(f'match found : {name}')
        else:
            name = 'Unknown'
            # print(f'match found : {name}')

        # draw rectangle around face
        y1, x2, y2, x1 = faceLoc
        # multiply by 4 to get original size
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        drawRect(frame, name, x1, y1, x2, y2)
        markAttendance(name)
        
def threadGrabAndProcessAndShow(source=0):
    video_grabber = VideoGrab(source).start()
    video_shower = VideoShow(video_grabber.frame).start()
    processor = FrameProcessor(video_grabber, video_shower, compareFaces).start()
    
    while True:
        if video_grabber.stopped or video_shower.stopped or processor.stopped:
            video_grabber.stop()
            video_shower.stop()
            processor.stop()
            break
        
# encoding process - find encodings for all images
def findEncodings(images):
    encodeList = []
    for img in images:
        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def drawRect(img, name, x1, y1, x2, y2):
    if name == 'Unknown':
        rectR, rectG, rectB = 83, 0, 0
        nameR, nameG, nameB = 254, 32, 32
    else:
        rectR, rectG, rectB = 0, 255, 127
        nameR, nameG, nameB = 255, 255, 255
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (rectB, rectG, rectR), 2)
    cv2.rectangle(img, (x1, y2-35), (x2, y2), (rectB, rectG, rectR), cv2.FILLED)
    cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (nameB, nameG, nameR), 2)
    
def markAttendance(name):
    def write_to_file():
        # TODO: expand functionality here, get creative
        with open('Logs/Attendance.csv', 'r+') as file:
            dataList = file.readlines()
            nameList = []
            for line in dataList:
                entry = line.split(',')
                # entry[0] is the name
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                file.writelines(f'\n{name},{dtString}')
    Thread(target=write_to_file).start()
        
encodeListKnownFaces = findEncodings(images)
print('Encoding Complete') # debug

threadGrabAndProcessAndShow(0)