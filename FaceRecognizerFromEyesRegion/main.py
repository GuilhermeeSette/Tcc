from FaceRecognizerFromEyesRegion.face_dataset import execute as capture_face
from FaceRecognizerFromEyesRegion.eye_roi_extractor import execute as extract_roi
from FaceRecognizerFromEyesRegion.align_faces import execute as align_face
from FaceRecognizerFromEyesRegion.face_resizer import resize as resize_image
import time


def execute():
    capture_face()
    time.sleep(5)
    align_face()
    time.sleep(5)
    extract_roi()
    time.sleep(5)
    resize_image()


if __name__ == '__main__':
    execute()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
