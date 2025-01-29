import cv2
import numpy as np
import face_recognition
import os

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
    for encodeFace, faceLoc in zip(encodeCurrFrame, facesCurrFrame):
        # compare current encodeFace to all known faces
        matches = face_recognition.compare_faces(encodeListKnownFaces, encodeFace)
        # find dist
        faceDis = face_recognition.face_distance(encodeListKnownFaces, encodeFace)
        print(faceDis)
        # set index of best match
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(f'match found : {name}')
            
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
        
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0, 245, 112), 2)
