import cv2
import numpy as np
import face_recognition

# import image
imgLeo = face_recognition.load_image_file('ImagesBasic/LeonardoDiCaprio.jpg')
# convert to RBG
imgLeo = cv2.cvtColor(imgLeo, cv2.COLOR_BGR2RGB)

# import test image
imgLeoTest = face_recognition.load_image_file('ImagesBasic/Leonardo_Test.jpg')
imgLeoTest = cv2.cvtColor(imgLeoTest, cv2.COLOR_BGR2RGB)

# locate faces
faceLoc = face_recognition.face_locations(imgLeo)[0]
encodeLeo = face_recognition.face_encodings(imgLeo)[0]

# draw rectangle around face
# faceLoc contains 4 values: top, right, bottom, left
# draw from top left to bottom right
cv2.rectangle(imgLeo, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 245, 112), 2)

faceLocTest = face_recognition.face_locations(imgLeoTest)[0]
encodeLeoTest = face_recognition.face_encodings(imgLeoTest)[0]
cv2.rectangle(imgLeoTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0, 245, 112), 2)

# comparing faces and finding the distance between them
# comparing the 128 encodings
results = face_recognition.compare_faces([encodeLeo], encodeLeoTest)
print(results[0])

# display images
cv2.imshow('Leo', imgLeo)
cv2.imshow('LeoTest', imgLeoTest)
cv2.waitKey(0)

