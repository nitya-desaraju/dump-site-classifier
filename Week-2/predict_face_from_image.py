from predict_faces import predict_faces
from overlay_box import overlay_box
from cos_dist import cos_dist
import os
import pickle
import numpy as np
from database_v1 import load_database

def predict_face_from_image(descriptors, boxes, image, cutoff):
    db_path = os.path.join(os.path.dirname(__file__), 'face_db.pkl')
    with open(db_path, 'rb') as file:
        db = pickle.load(file)

    db_descriptors = [profile.average_descriptor() for profile in db.values()]
    db_descriptors = np.array(db_descriptors)

    distances = cos_dist(descriptors, db_descriptors)
    names = predict_faces(distances, descriptors, cutoff)
    overlay_box(image, names, boxes)
    