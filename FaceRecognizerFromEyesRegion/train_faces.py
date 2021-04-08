import cv2
from calculator import *

def train(faces, test_face):
  mean_face = equation2(faces)
  differences_between_faces_mean_face = equation3(faces, mean_face)
  eigenvectors = equation5(differences_between_faces_mean_face)
  hotelling_transform = equation6(differences_between_faces_mean_face, eigenvectors)
  projection_weigths = equation7(hotelling_transform, faces, mean_face)
  reconstructed_face = equation8(hotelling_transform, projection_weigths, mean_face)
  reconstruction_error = equation9(faces, test_face)
  close_error_difference_k = equation10(test_face, faces)
  return [reconstruction_error, close_error_difference_k]