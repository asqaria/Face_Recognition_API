from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from create_db import Base, Users, Visitors, Suspicious
import dlib
import cv2
import openface
from PIL import Image
import numpy as np
import os

engine = create_engine("sqlite:///database.db")
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = scoped_session(DBSession)

app = Flask(__name__)

# Pre-trained face detection model here:
predictor_model = "face_landmark.dat"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_aligner = openface.AlignDlib(predictor_model)

# Allowed extensions for images
ALLOWED_EXTENSIONS = set(['png', 'jpg'])
# Minimum amount of images
MIN_IMG = 2

# model='nn4.small2.def.lua'
net = openface.TorchNeuralNet(model='nn4.small2.v1.t7', imgDim=96, cuda=False)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/create', methods=['POST'])
def create():
    name = request.form['name']
    email = request.form['email']
    phone = request.form['phone']
    picture = request.files.get('picture')
    images = request.files.getlist('images')

    if name == '' or email == '' or phone == '' or len(images) < MIN_IMG or not picture:
        return 'You are required to fill up all forms!'

    total_images = len(images)
    total_faces = 0

    for i in images:
        # Run the HOG face detector for all images.
        image = cv2.cvtColor(np.array(Image.open(i.stream)), cv2.COLOR_BGR2RGB)
        detected_faces = face_detector(image, 1)
        total_faces += len(detected_faces)

        matrix = list()

        # Crop and align all faces
        for j, face_rect in enumerate(detected_faces):
            aligned_face = face_aligner.align(96, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            matrix.append(net.forward(aligned_face))

    if total_faces < MIN_IMG or total_images < MIN_IMG:
        return 'Required minimum %d face images per user' % MIN_IMG

    if picture and allowed_file(picture.filename):
        extension = picture.filename.split('.')[1]
        user = Users(name=name, email=email, phone=phone, image=extension)
        session.add(user)
        session.commit()
        filename = 'picture.%s' % extension
        path = 'images/users/%s' % user.id
        os.mkdir(path)
        picture.save(os.path.join(path, filename))
    else:
        return 'Error while uploading picture'

    result = {
        'id': user.id,
        'images': total_images,
        'detected_faces': total_faces
    }

    return jsonify(result)


if __name__ == "__main__":
    app.secret_key = 'super_secret_key'
    app.debug = True
    app.run(host='0.0.0.0', port=3434)
