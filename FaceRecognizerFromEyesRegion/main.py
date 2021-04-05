from FaceRecognizerFromEyesRegion.face_dataset import execute as capture_face
from FaceRecognizerFromEyesRegion.eye_roi_extractor import execute as extract_roi
from FaceRecognizerFromEyesRegion.align_faces import execute as align_face
from FaceRecognizerFromEyesRegion.train_faces import train as train_faces
from FaceRecognizerFromEyesRegion.face_resizer import resize as resize_image
from FaceRecognizerFromEyesRegion.recognizer import recognize as recognize
import time
import glob2


def execute():
    capture_face()
    time.sleep(5)
    align_face()
    time.sleep(5)
    extract_roi()
    time.sleep(5)
    resize_image()
    train_faces = [cv2.imread(file) for file in glob2.glob("./*.png")]
    test_face = [cv2.imread('FaceRecognizerFromEyesRegion/train_face.png')]
    train_faces_projection_weights = train_faces(train_faces)
    test_face_projection_weights = train_faces(test_face)
    recognize(train_faces_projection_weights, test_face_projection_weights)


if __name__ == '__main__':
    execute()

