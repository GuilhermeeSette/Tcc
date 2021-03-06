import cv2
import time


def execute():
    name = input("Digite o nome do dono da face: ")
    cam = cv2.VideoCapture(1)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0

    while True:

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            cv2.imshow('image', img)
        time.sleep(3)
        #cv2.imwrite("./images/%s.png" % name, img)
        cv2.imwrite("./%s.png" % name, img)
        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            cam.release()
            cv2.destroyAllWindows()
            return f'{name}.png'
            break
        elif count >= 30:  # Take 30 face sample and stop video
            cam.release()
            cv2.destroyAllWindows()
            return f'{name}.png'
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
