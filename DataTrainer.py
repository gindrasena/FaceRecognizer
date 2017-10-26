"Author:Indrasena Reddy G"
"Import the Packages"
import cv2 as cv
import numpy as np
import os


print("*************************")
print("Welcome to QCOM Eye")
print("*************************")
employee_Id = input("Enter Employee Number:")
if (employee_Id is None):
    exit()
faceClassifier=cv.CascadeClassifier('TrainerClues\haarcascade_frontalface_default.xml')
os.chdir('DataSet')
check=os.mkdir(employee_Id,777)
os.chdir(str(employee_Id))
capture=cv.VideoCapture(0)
sample=0
while (True):
    ret,frame=capture.read()
    gray_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face_frame=faceClassifier.detectMultiScale(gray_frame,1.3,5)
    for (x,y,w,h) in face_frame:
        sample=sample+1
        cv.imwrite(str(sample)+".jpg",gray_frame[y:y+h,x:x+w])
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0,),2)
        cv.waitKey(50)
        
    cv.waitKey(1)
    if(sample>20):
        break
capture.release()
cv.destroyAllWindows()
