import cv2
import numpy as np
import face_recognition
import os

# defining vars
path = 'ImagesAttendance'
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
# print(len(encodeListKnownFaces)) # debug        
        
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0, 245, 112), 2)
