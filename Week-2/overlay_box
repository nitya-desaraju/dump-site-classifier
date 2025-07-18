import matplotlib.pyplot as plt
import numpy as np

#takes in image array, array of names, array of boxes
def overlay_box(image, names, boxes):
    #if only one name, makes sure it's in list
    if isinstance(names, list):
        pass
    else:
        names = [names]
        
    for i in range(len(boxes)):

        #setting up the boxes
        b = boxes[i]
        n = names[i]
        x_min = b[0]
        y_min = b[1]
        x_max = b[2]
        y_max = b[3]

        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)

        plt.plot(x, np.ones(100) * y_min, color='red')
        plt.plot(x, np.ones(100) * y_max, color='red')
        plt.plot(np.ones(100) * x_min, y, color='red')
        plt.plot(np.ones(100) * x_max, y, color='red')

        #putting the name
        plt.text((x_min + x_max)/2, y_max + 35, n, color='white')
        
    #loading the image
    plt.imshow(image)

    #saving the file
    file_name = ", ".join(names) + " detected faces.jpg"
    plt.savefig(file_name, dpi=600) 

    #showing the plot
    plt.show()

    
