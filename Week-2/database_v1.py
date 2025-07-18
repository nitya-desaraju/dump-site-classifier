import numpy as np
import pickle

#This class creates a profile for each name
class Profile:
    #This initializes the profile class
    def __init__(self, name):
        self.name = name
        self.descriptors = [] 
        
    #This allows a descriptor to be added for a person
    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)
        
    #This averages the descriptors for a person
    def average_descriptor(self):
        return np.mean(self.descriptors, axis=0)

#This function creates the database
def create_database(image_list, name_list, img_to_descriptors_func, filename="face_db.pkl"):
    #Create empty database
    database = {}

    #Iterate through input images and names
    for img, name in zip(image_list, name_list):
        descriptors, _ = img_to_descriptors_func(img)

        #Add name to database if necessary
        if name not in database:
            database[name] = Profile(name)

        #Add descriptor to name if the name is already present
        for d in descriptors:
            database[name].add_descriptor(d)

    #Open the database
    with open(filename, 'wb') as f:
        pickle.dump(database, f)

    print(f"Database saved to {filename}")

#Save the database
def save_database(database, filename="face_db.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(database, f)

#Load the database
def load_database(filename="face_db.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)

#Add image to already existing database
def add_picture_to_database(image, name, img_to_descriptors_func, db_file="face_db.pkl"):
    #Load the existing database
    try:
        database = load_database(db_file)
    except FileNotFoundError:
        database = {}

    #Extract descriptors from the new image
    descriptors, _ = img_to_descriptors_func(image)

    #Add descriptors to the database
    if name not in database:
        database[name] = Profile(name)

    for descriptor in descriptors:
        database[name].add_descriptor(descriptor)

    #Save the updated database
    save_database(database, db_file)

    print(f"Added new picture for {name} and saved the updated database.")
 