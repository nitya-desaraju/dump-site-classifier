import numpy as np
import pickle
import os
from database_v1 import Profile #importing Profile class

def predict_faces(distances, descriptors, cutoff):
    """
    Predicts face matches based on cosine distances and updates database

    Parameters:
    - distances: np.ndarray, shape=(M,N), distances to each person
    - descriptors: descriptor vectors of the faces being compared
    - cutoff: float, max distance to match

    Returns:
    - name of best match or name inputted by user, updates database
    """

    #load database
    db_path = os.path.join(os.path.dirname(__file__), 'face_db.pkl') 
    with open(db_path, 'rb') as file:
        db = pickle.load(file)
        

    names = list(db.keys())
    faces = distances.shape[1]
    results = []

    #find min distance
    for i in range(faces):
        face_distances = distances[:, i]  
        descriptor = descriptors[i]   

        min_dist = np.min(face_distances)
        idx = np.argmin(face_distances)
        match = names[idx]


        #add to database based on threshold
        if min_dist < cutoff:
            print(f"[Face {i+1}] matched with {match}!")
            db[match].add_descriptor(descriptor)
            results.append(match)

        else:
            print(f"[Face {i+1}] is unknown.")
            name = input("Enter name for this person: ").strip()

            if name in db:
                print(f"Adding descriptor to {name}'s profile.")
                db[name].add_descriptor(descriptor)

            else:
                print(f"Creating new profile for: {name}")
                new_profile = Profile(name)
                new_profile.add_descriptor(descriptor)
                db[name] = new_profile

            results.append(name)

    #save database
    with open(db_path, 'wb') as file:
        pickle.dump(db, file)

    return results