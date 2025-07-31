import numpy as np
import pickle

def generate_dataset(labels_pkl : str, heatmaps_pkl : str, images_pkl : str, size, k=1, dist=-1, num_classes=22):
    '''
    labels_pkl file holds an array of the corresponding classes for images in images_pkl
    Note: if we want to make num_classes<22, then we need to remap the labels to numbers 
    less than num_classes before pushing labels_pkl into generate_dataset

    Returns: tuple of four np.ndarrays x_train, y_train, x_test, and y_test
    '''
    
    
    with open(heatmaps_pkl, 'rb') as h:
        heatmaps = pickle.load(h)
    with open(images_pkl, 'rb') as img:
        images = pickle.load(img)
    with open(labels_pkl, 'rb') as l:
        labels = pickle.load(l)
    
    x = []
    y = []
    
    for i in range(heatmaps.shape[0]):
        points = identify_dumps(heatmaps[i], size, k, dist)
        for p in points:
            boxes = drawing_bounding_boxes(size[0], size[1], p[0], p[1], images[i])
            x.append(box)
            one_hot = np.zeros(num_classes)
            one_hot[labels[i]-1]=1 # THIS IS ASSUMING LABELS ARE 1-22, NOT 0-21
            y.extend([one_hot for j in boxes])

    #Making the train:test ratio 9:1 (num of training samples is more important than validation)
    x_train = np.array(x[:9*len(x)//10])
    y_train = np.array(y[:9*len(y)//10])
    x_test = np.array(x[9*len(x)//10:])
    y_test = np.array(y[9*len(y)//10:])

    with open("x_train.pkl", 'wb') as xtrn:
        pickle.dump(x_train, xtrn)
    with open("x_test.pkl", 'wb') as xtst:
        pickle.dump(x_test, xtst)
    with open("y_train.pkl", 'wb') as ytrn:
        pickle.dump(y_train, ytrn)
    with open("y_test.pkl", 'wb') as ytst:
        pickle.dump(y_test, ytst)
    
    return (x_train, y_train, x_test, y_test)
        
