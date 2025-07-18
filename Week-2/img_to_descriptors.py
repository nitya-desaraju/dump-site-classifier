from facenet_models import FacenetModel
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def img_to_descriptors(pic):
    model = FacenetModel()
    boxes, probabilities, landmarks = model.detect(pic)

    #This function checks if an image is below the probability theshold.
    def filter_false_positive(image, boxes, probs, threshold=0.9):
        #Converts boxes and probabilities into arrays if needed
        boxes = np.array(boxes)
        probs = np.array(probs)
        
        #Check if probability is above theshold
        keep = probs > threshold
        filtered_boxes = boxes[keep]
        return image, filtered_boxes

    _, filtered_boxes = filter_false_positive(pic, boxes, probabilities)
    descriptors = model.compute_descriptors(pic, filtered_boxes)
    
    return descriptors, filtered_boxes
