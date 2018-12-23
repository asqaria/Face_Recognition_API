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
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


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

    if picture and allowed_file(picture.filename):
        extension = picture.filename.split('.')[1]
        user = Users(name=name, email=email, phone=phone, image=extension)
        session.add(user)
        session.commit()
        filename = '%s.%s' % (user.id, extension)
        picture.save(os.path.join('images/users', filename))
    else:
        return 'Error while uploading picture'

    for i in images:
        # Run the HOG face detector for all images.
        image = cv2.cvtColor(np.array(Image.open(i.stream)), cv2.COLOR_BGR2RGB)
        detected_faces = face_detector(image, 1)

        aligned_faces = list()

        # Crop and align all faces
        for j, face_rect in enumerate(detected_faces):
            aligned_face = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            aligned_faces.append(aligned_face)

    return jsonify(Users=[user.serialize])


if __name__ == "__main__":
    app.secret_key = 'super_secret_key'
    app.debug = True
    app.run(host='0.0.0.0', port=3434)
