import cv2, openface, dlib, numpy as np
from PIL import Image

# Pre-trained face detection model here:
predictor_model = "features/face_landmark.dat"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_aligner = openface.AlignDlib(predictor_model)
modeldir = 'features/nn4.small2.v1.t7'

net = openface.TorchNeuralNet(model=modeldir, imgDim=96, cuda=False)


def get_rep(images):
    matrices = list()
    total_faces = 0

    # Run the HOG face detector for all images.
    image = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(image, 1)
    total_faces += len(detected_faces)

    # Crop and align all faces
    for j, face_rect in enumerate(detected_faces):
        aligned_face = face_aligner.align(96, image, face_rect,
                                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        matrices.append(net.forward(aligned_face))
        cv2.imshow('img', aligned_face)
    return total_faces, matrices


images = cv2.imread('25.jpg')
get_rep(images)

cv2.waitKey(0)
cv2.destroyAllWindows()