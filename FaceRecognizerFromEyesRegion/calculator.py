import cv2
import numpy as np


#Calculate the mean face
def equation2(images, number_of_faces):
    M = number_of_faces  # number of training faces
    img1 = cv2.imread(images[0])
    img2 = cv2.imread(images[1])
    gama = [img1, img2]
    mean_face = 1 / M * np.sum(gama)
    return mean_face


#Vector with differences between each face and mean face
def equation3(faces, mean_face):
    differences_between_faces_mean_face = []
    for i in faces:
        difference = i - mean_face
        differences_between_faces_mean_face.append(difference)
    return differences_between_faces_mean_face


#Eigenvector and Eigenvalues
def equation5(differences_between_faces_mean_face):
    transposed_differences_between_faces_mean_face = np.transpose(differences_between_faces_mean_face)
    return differences_between_faces_mean_face * transposed_differences_between_faces_mean_face


#Hotelling transform
def equation6(differences_between_faces_mean_face, eigen_vectors_and_values):
    return differences_between_faces_mean_face * eigen_vectors_and_values


#K-weights of an image ( projection_weights )
def equation7(hotelling_transform,faces, mean_face):
    omega_weights = []
    transposed_hotelling_transform = np.transpose(hotelling_transform)
    for face in faces:
        omega_weight = transposed_hotelling_transform * (face - mean_face)
        omega_weights.append(omega_weight)
    return omega_weight


## Recognition Phase


#Reconstructed face
def equation8(hotelling_transform, projection_weights, mean_face):
    return hotelling_transform * projection_weights + mean_face


#Reconstruction error between the face and its reconstruction
def equation9(face, reconstructed_face):
    return abs(face - reconstructed_face)


#Minimum distance test_face and faces stored using K-NN technique
def equation10(test_face, faces):
    errors = []
    for face in faces:
        error = test_face - face
        errors.append(error)
    return np.min(errors)