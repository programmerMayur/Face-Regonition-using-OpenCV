import cv2
import numpy as np
import face_recognition

# load and convert image to RGB
imgModi = face_recognition.load_image_file('img/Narendra Modi.jpg')
imgModi = cv2.cvtColor(imgModi,cv2.COLOR_BGR2RGB)

# load and convert test image to RGB
imgTest = face_recognition.load_image_file('img/imgTest.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#Face Locate and Encoding
faceLoc = face_recognition.face_locations(imgModi)[0]
encodeModi = face_recognition.face_encodings(imgModi)[0]
cv2.rectangle(imgModi,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoc = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#Compare the faces
result = face_recognition.compare_faces([encodeModi],encodeTest)
faceDis = face_recognition.face_distance([encodeModi],encodeTest)
matchOnly = 100 - (faceDis * 100)
matchOnly = round(matchOnly[0], 2)
print(result)
print(faceDis)

cv2.putText(imgTest,f"{result} {round(faceDis[0],2)} Match:{matchOnly}%",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Narendra Modi',imgModi)
cv2.imshow('Test Narendra Modi',imgTest)

cv2.waitKey(0)