import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

# TODO: implement multithreading to increase performance

# defining vars
path = 'ImagesWebcam'
images = []
classNames = []
files = os.listdir(path)
print(files) # debug

# loop through files
# append images and classNames
for fileName in files:
    currImg = cv2.imread(f'{path}/{fileName}')
    images.append(currImg)
    classNames.append(os.path.splitext(fileName)[0].replace('_', ' '))

print(classNames) # debug

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
        
encodeListKnownFaces = findEncodings(images)
print('Encoding Complete') # debug

# third step - find matching between faces
# coming from webcam
cap = cv2.VideoCapture(0)

# get each frame
while True:
    success, img = cap.read()
    # reduce size of image
    imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    # convert to RGB
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    
    # find encoding of webcam's current frame
    facesCurrFrame = face_recognition.face_locations(imgSmall)
    encodeCurrFrame = face_recognition.face_encodings(imgSmall, facesCurrFrame)
    
    # finding matches
    # compare all faces to all known faces (encodings)
    # TODO: extract this to a function
    for encodeFace, faceLoc in zip(encodeCurrFrame, facesCurrFrame):
        # compare current encodeFace to all known faces
        matches = face_recognition.compare_faces(encodeListKnownFaces, encodeFace)
        # find dist
        faceDis = face_recognition.face_distance(encodeListKnownFaces, encodeFace)
        print(faceDis)
        # set index of best match
        matchIndex = np.argmin(faceDis)
        
        # print the name of the match if it exists
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)
            # print(f'match found : {name}')
        else:
            name = 'Unknown'
            markAttendance(name)
            # print(f'match found : {name}')

        # draw rectangle around face
        y1, x2, y2, x1 = faceLoc
        # multiply by 4 to get original size
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        drawRect(img, name, x1, y1, x2, y2)
            
    cv2.imshow('webcam', img)
    # quit on "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            