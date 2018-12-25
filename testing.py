from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from operator import itemgetter
import argparse
import cv2
import os
import pickle
import sys
import pandas as pd
import openface
import time
import numpy as np
import csv
from PIL import Image
import dlib


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

