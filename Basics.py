import cv2
import numpy as np
import face_recognition

# import image
imgLeo = face_recognition.load_image_file('ImagesBasic/LeonardoDiCaprio.jpg')
# convert to RBG
imgLeo = cv2.cvtColor(imgLeo, cv2.COLOR_BGR2RGB)

# import test image
# imgTest = face_recognition.load_image_file('ImagesBasic/Leonardo_Test.jpg')
imgTest = face_recognition.load_image_file('ImagesBasic/BarackObama.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# locate faces
faceLoc = face_recognition.face_locations(imgLeo)[0]
encodeLeo = face_recognition.face_encodings(imgLeo)[0]

# draw rectangle around face
# faceLoc contains 4 values: top, right, bottom, left
# draw from top left to bottom right
cv2.rectangle(imgLeo, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 245, 112), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0, 245, 112), 2)

# comparing faces and finding the distance between them
# comparing the 128 encodings
results = face_recognition.compare_faces([encodeLeo], encodeTest)
# the lower the dist, the better the match
faceDis = face_recognition.face_distance([encodeLeo], encodeTest)
print(results[0], faceDis)
# cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,200),2)
if faceDis < 0.6:
    cv2.putText(imgTest, f'{results[0]} {round(faceDis[0],2)}', (faceLocTest[3], faceLocTest[2]), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,0),2)
else:
    cv2.putText(imgTest, f'{results[0]} {round(faceDis[0],2)}', (faceLocTest[3], faceLocTest[2]), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),2)

# display images
cv2.imshow('Leo', imgLeo)
cv2.imshow('LeoTest', imgTest)
cv2.waitKey(0)

