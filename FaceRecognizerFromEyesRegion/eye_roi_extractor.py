import numpy as np
import cv2

# global var:
eye_counter = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(1)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #define regiao de interesse em cinza e em cor
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            y = int(0.5 * h)
            y2 = int(y / 2.5)
            # here's your eye-roi, see, it's the very same pattern
            #define regiao dos olhos de interesse
            roi_color_eye = roi_color[y2:(y-y2)+eh, 0:w]
            # write image *before* drawing stuff on it
            cv2.imwrite("eye_region.png" % eye_counter, roi_color_eye)

            eye_counter += 1

            #desenha retangulo do roi dos olhos
            cv2.rectangle(roi_color, (0, y), (w, y2), (0, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

crop_img = roi_color
cv2.imwrite("path", roi_color)