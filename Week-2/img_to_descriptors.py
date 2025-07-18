from facenet_models import FacenetModel
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def img_to_descriptors(pic, threshold):
    model = FacenetModel()
    boxes, probabilities, landmarks = model.detect(pic)

    assert probabilities is not None, "No face detected"

    #This function checks if an image is below the probability theshold.
    def filter_false_positive(image, boxes, probs, threshold):
        #Converts boxes and probabilities into arrays if needed
        boxes = np.array(boxes)
        probs = np.array(probs)

        if probs[0] is None:
            return -1, -1
            
        #Check if probability is above theshold
        keep = probs > threshold
        filtered_boxes = boxes[keep]
        return image, filtered_boxes

    _, filtered_boxes = filter_false_positive(pic, boxes, probabilities, threshold)
    if filtered_boxes == -1:
        return -1, -1
    descriptors = model.compute_descriptors(pic, filtered_boxes)
    
    return descriptors, filtered_boxes
