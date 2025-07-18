import numpy as np

#This function checks if an image is below the probability theshold.

def filter_false_positive(image, boxes, probs, threshold=0.9):
    #Converts boxes and probabilities into arrays if needed
    boxes = np.array(boxes)
    probs = np.array(probs)
    
    #Check if probability is above theshold
    keep = probs > threshold
    filtered_boxes = boxes[keep]
    return image, filtered_boxes