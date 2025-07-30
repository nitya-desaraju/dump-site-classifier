def draw_bounding_boxes(height, width, x, y, image, show_graph=False):
    x_min = int(x-width/2)
    x_max = int(x+width/2)
    y_min = int(y-height/2)
    y_max = int(y+height/2)
    print(x_min, x_max, y_min, y_max)
    cropped_image = image[y_min : y_max, x_min: x_max]
    results = np.zeros((4, height, width, 3), dtype=int)

    for i in range(4):
        cropped_image = np.rot90(cropped_image)
        results[i] = cropped_image

    if show_graph:
        x_line = np.linspace(x_min, x_max, 100)
        y_line = np.linspace(y_min, y_max, 100)
    
        plt.plot(x_line, np.ones(100) * y_min, color='red')
        plt.plot(x_line, np.ones(100) * y_max, color='red')
        plt.plot(np.ones(100) * x_min, y_line, color='red')
        plt.plot(np.ones(100) * x_max, y_line, color='red')
    
        plt.imshow(image)
    
        file_name = "figure " + str(random.randint(0, 1000)) + ".jpg"
        plt.savefig(file_name, dpi=600) 

        plt.show()

    return results
