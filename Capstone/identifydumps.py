import numpy as np

def identify_dumps(image, size, k=1, dist=-1):
    '''
    image: a heatmap np.ndarray of shape (H,W)
    size: size of the bounding box
    '''
    if dist==-1:
        dist = max(size[0]//2, size[1]//2)
    
    pfx_sums = np.zeros(np.array(image.shape)+np.array([1,1,1]))
    
    for i,x in enumerate(image):
        for j,y in enumerate(x):
            pfx_sums[i+1,j+1] = image[i,j] + pfx_sums[i+1,j] + pfx_sums[i,j+1] - pfx_sums[i,j]

    sorted_points = []
    
    for i in range(len(size[0]-1, image.shape[0])):
        for j in range(len(size[1]-1, image.shape[1])):
            count = pfx_sums[i+1, j+1] - pfx_sums[i+1-size[0], j+1] - pfx_sums[i+1, j+1-size[1]] + pfx_sums[i+1-size[0], j+1-size[1]]
            sorted_points.append( (count , np.array([i-(size[0]-1)//2,j-(size[1]-1)//2]) ) )
    sorted_points.sort(key=lambda x:-x[0])
    
    points = []

    for p in sorted_points:
        add = True
        for x in points:
            d = np.linalg.norm(p[1]-x)
            add = (d>=dist)
            if not add:
                break
        if add:
            points.append(p[1])

    return np.array(points)
    
    
