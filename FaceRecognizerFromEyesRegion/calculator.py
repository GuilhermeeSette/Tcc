import cv2
import numpy as np
from PIL import Image

#Calculate the mean face
def equation2(images):
    M = len(images)  # number of training faces
    gama = []
    sum = 0
    #Vector of pattern images
    for i in range(0,M):
        gama.append(images[i])
    #Sommatory of pattern images
    for i in gama:
        sum = sum  + i
    mean_face = 1 / M * sum
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
    differences_between_faces_mean_face = np.array(differences_between_faces_mean_face)
    
    # transposed_differences_between_faces_mean_face = np.transpose(differences_between_faces_mean_face)

    return differences_between_faces_mean_face * differences_between_faces_mean_face.T


#Hotelling transform
def equation6(differences_between_faces_mean_face, eigen_vectors_and_values):
    return differences_between_faces_mean_face * eigen_vectors_and_values


#K-weights of an image ( projection_weights )
def equation7(hotelling_transform,faces, mean_face):
    omega_weights = []
    transposed_hotelling_transform = np.transpose(hotelling_transform)
    for face in faces:
        omega_weight = transposed_hotelling_transform * np.array((face - mean_face))
        omega_weights.append(omega_weight)
    return omega_weights


## Recognition Phase


#Reconstructed face
def equation8(hotelling_transform, projection_weights, mean_face):
    return hotelling_transform * projection_weights + mean_face


#Reconstruction error between the face and its reconstruction
def equation9(face, reconstructed_face):
    return abs(np.linalg.norm((face - np.array(reconstructed_face))))


#Minimum distance test_face and faces stored using K-NN technique
def equation10(test_face, faces):
    errors = []
    for face in faces:
        error = test_face - face
        errors.append(np.linalg.norm(error))
    #index, min_error
    return [errors.index(np.min(errors)), np.min(errors)]

# img1 = cv2.imread('face.png')
# img2 = cv2.imread('face2.png')
# equation2([img1,img2], 2)