from face_dataset import execute as capture_face
from eye_roi_extractor import execute as extract_roi
from align_faces import execute as align_face
from train_faces import train as train_faces
from face_resizer import resize as resize_image
from recognizer import recognize as recognize
from database import db
import time
import glob2
import cv2

database = db()
def execute():
    # face = capture_face()
    face = "luizk.png"
    time.sleep(5)
    align_face(face)
    time.sleep(5)
    extract_roi(face)
    # time.sleep(5)
    # resize_image(face)
    # faces_to_be_trained = [cv2.imread(file) for file in glob2.glob("./images/*.png")]
    # test_face = cv2.imread('./test_face.png')
    # error_values = train_faces(faces_to_be_trained, test_face)
    # print(database[recognize(error_values)])


if __name__ == '__main__':
    execute()

