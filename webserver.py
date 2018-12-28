from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from create_db import Base, Users, Visitors, Suspicious
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import dlib
import cv2
import openface
from PIL import Image
import numpy as np
import csv
import pickle
import sys
import os
import pandas as pd

engine = create_engine("sqlite:///database.db")
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = scoped_session(DBSession)

app = Flask(__name__)

# Pre-trained face detection model here:
predictor_model = "features/face_landmark.dat"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_aligner = openface.AlignDlib(predictor_model)

# Allowed extensions for images
ALLOWED_EXTENSIONS = set(['png', 'jpg'])

# Minimum amount of images
MIN_IMG = 2

# Minimum confidence
TRESHOLD = 0.85

# Deep network installation
modeldir = 'features/nn4.small2.v1.t7'
net = openface.TorchNeuralNet(model=modeldir, imgDim=96, cuda=False)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_rep(images):
    matrices = list()
    total_faces = 0

    for i in images:
        # Run the HOG face detector for all images.
        image = cv2.cvtColor(np.array(Image.open(i.stream)), cv2.COLOR_BGR2RGB)
        detected_faces = face_detector(image, 1)
        total_faces += len(detected_faces)

        # Crop and align all faces
        for j, face_rect in enumerate(detected_faces):
            aligned_face = face_aligner.align(96, image, face_rect,
                                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            matrices.append(net.forward(aligned_face))
    return total_faces, matrices


def train(dir='images/users', workdir='features'):
    labels = list()
    merged_csv_path = os.path.join(workdir, 'merged.csv')
    merged = open(merged_csv_path, "w+")

    for filename in sorted(os.listdir(dir)):
        new_dir = os.path.join(dir, filename)
        for csv in sorted(os.listdir(new_dir)):
            if csv.endswith('.csv'):
                labels.append(filename)
                fullpath = os.path.join(dir, filename, csv)
                fl = open(fullpath)
                for line in fl:
                    merged.write(line)
                fl.close()
    merged.close()

    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    embeddings = pd.read_csv(merged_csv_path, header=None).as_matrix()

    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(workdir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)


@app.route('/recognize/', methods=['POST'])
def recognize():
    images = request.files.getlist('images')
    total_faces, matrices = get_rep(images)
    classifier = 'features/classifier.pkl'

    with open(classifier, 'rb') as f:
        if sys.version_info[0] < 3:
            (le, clf) = pickle.load(f)
        else:
            (le, clf) = pickle.load(f, encoding='latin1')

    labels = list()
    names = list()
    confidences = list()

    for r in matrices:
        predictions = clf.predict_proba(r).ravel()
        maxI = np.argmax(predictions)

        id = int(le.inverse_transform(maxI).decode('utf-8'))
        user = session.query(Users).filter_by(id=id).one()
        labels.append(id)
        names.append(user.name)
        confidences.append(predictions[maxI])

    if len(confidences) > 0 and max(confidences) > TRESHOLD:
        best_idx = confidences.index(max(confidences))
        return jsonify({
            'id': labels[best_idx],
            'name': names[best_idx],
            'confidence': confidences[best_idx]
        })
    else:
        return jsonify({
            'id': 'Null',
            'name': 'Null',
            'confidence': 'Null'
        })


@app.route('/create/', methods=['POST, GET'])
def create():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        picture = request.files.get('picture')
        images = request.files.getlist('images')

        if name == '' or email == '' or phone == '' or len(images) < MIN_IMG or not picture:
            return 'You are required to fill up all forms!'

        total_images = len(images)
        total_faces, matrices = get_rep(images)

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

            # create matrix for each face
            index = 0
            for matrix in matrices:
                filename = '%s.csv' % index
                path = os.path.join('images/users/%s' % user.id, filename)
                with open(path, 'w+') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(matrix)
                index += 1
        else:
            return 'Error while uploading picture'

        result = {
            'id': user.id,
            'images': total_images,
            'detected_faces': total_faces
        }

        # update classifier.pkl
        train()
        return jsonify(result)
    elif request.method == 'GET':
        return render_template('register.html')


@app.route('/update/', methods=['POST'])
def update():
    pass


@app.route('/delete/', methods=['POST'])
def delete():
    pass


@app.route('/', methods=['GET'])
def main():
    pass


@app.route('/visitors/', methods=['GET'])
def visitors():
    pass


@app.route('/suspicious/', methods=['GET'])
def suspicious():
    pass


@app.route('/user/<int:id>/', methods=['GET'])
def user_id(id):
    pass


@app.route('/suspicious/<int:id>', methods=['GET'])
def suspicious_id(id):
    pass


if __name__ == "__main__":
    app.secret_key = 'super_secret_key'
    app.debug = True
    app.run(host='0.0.0.0', port=3434)
