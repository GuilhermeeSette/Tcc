import numpy as np
import cv2

# global var:

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


# cap = cv2.VideoCapture(1)


def execute(image_path):
    eye_counter = 0
    # img = cv2.imread("./images/%s" % image_path)
    img = cv2.imread("./%s" % image_path)
    while 1 and eye_counter < 10:
        # ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                y = int(0.5 * h)
                height = int(y / 2.5)
                # Get roi through python array slicing
                roi_color_eye = roi_color[height:(y - height) + eh, 0:w]
                cv2.imwrite("./%s" % image_path, roi_color_eye)
                if eye_counter == 10:
                    quit()
                eye_counter += 1
            if eye_counter == 10:
                break


                # if eye_counter == 10:
                #     break

                # desenha retangulo do roi dos olhos
                # cv2.rectangle(roi_color, (0, y), (w, y2), (0, 255, 0), 2)
        if eye_counter == 10:
            break

        # cv2.imshow('img', img)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break
    #
    # # cap.release()
    # cv2.destroyAllWindows()
    #
    # crop_img = roi_color
    # cv2.imwrite("path", roi_color)
