import cv2


def resize():
    image = cv2.imread('FaceRecognizerFromEyesRegion/face.png')
    image_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    cv2.imwrite("FaceRecognizerFromEyesRegion/face.png", image_resized)
