import numpy as np
bitmasks = None # prefix sum array

def maps2masks(heatmaps, threshold=0.25):
    global bitmasks
    bitmasks = np.zeros(np.array(heatmaps.shape)+np.array([0,0,1,1]))
    
    for i,x in enumerate(heatmaps):
        for j,y in enumerate(x[0]):
            for k,z in enumerate(y):
                if z<=0.25:
                    bitmasks[i,0,j+1,k+1] = 1
                bitmasks[i,0,j+1,k+1] += bitmasks[i,0,j,k+1] + bitmasks[i,0,j+1,k] - bitmasks[i,0,j,k]
    
    bitmasks.reshape((bitmasks.shape[0],bitmasks.shape[2],bitmasks.shape[3]))
    return

def optimal_bounding_box(heatmaps : np.ndarray, fitting=0.0, threshold=0.25):
    '''
    fitting represents the maximum allowed percentage (0.0-1.0) of 
    below-threshold (0.25) heat values within the bounding box
    '''
    
    maps2masks(heatmaps, threshold)

    lo_r = 10
    hi_r = heatmaps.shape[2]//4
    
    i=0
    #FINDING THE OPTIMAL SQUARE BONDING BOX
    while lo_r < hi_r:
        mid = (lo_r + hi_r)//2
        j=i
        while j<heatmaps.shape[0]:
            x,y = np.array(identify_dump(heatmaps[i,0],(2*mid+1,2*mid+1))) + np.array([1,1])
            count = bitmasks[x+mid,y+mid] - bitmasks[x-mid-1,y+mid] - bitmasks[x+mid,y-mid-1] + bitmasks[x-mid-1,y-mid-1]
            if not (count/(2*mid+1)**2<=fitting):
                break
            j+=1
        if j<heatmaps.shape[0]:
            i=j
            lo_r = mid+1
        else:
            hi_r = mid

    lo_w = lo_r
    hi_w = heatmaps.shape[2]//4

    #OPTIMAL BOUNDING BOX IF SQUARE STRETCHED HORIZONTALLY
    i=0
    while lo_w < hi_w:
        mid = (lo_w + hi_w)//2
        j=i
        while j<heatmaps.shape[0]:
            x,y = np.array(identify_dump(heatmaps[i,0],(2*lo_r+1,2*mid+1))) + np.array([1,1])
            count = bitmasks[x+lo_r,y+mid] - bitmasks[x-lo_r-1,y+mid] - bitmasks[x+lo_r,y-mid-1] + bitmasks[x-lo_r-1,y-mid-1]
            if not (count/((2*lo_r+1)*(2*mid+1))**2<=fitting):
                break
            j+=1
        if j<heatmaps.shape[0]:
            i=j
            lo_w = mid+1
        else:
            hi_w = mid
    
    lo_h = lo_r
    hi_h = heatmaps.shape[2]//4
    #OPTIMAL BOUNDING BOX IF SQUARE STRETCHED VERTICALLY
    i=0
    while lo_h < hi_h:
        mid = (lo_h + hi_h)//2
        j=i
        while j<heatmaps.shape[0]:
            x,y = np.array(identify_dump(heatmaps[i,0],(2*mid+1,2*lo_r+1))) + np.array([1,1])
            count = bitmasks[x+mid,y+lo_r] - bitmasks[x-mid-1,y+lo_r] - bitmasks[x+mid,y-lo_r-1] + bitmasks[x-mid-1,y-lo_r-1]
            if not (count/((2*mid+1)*(2*lo_r+1))**2<=fitting):
                break
            j+=1
        if j<heatmaps.shape[0]:
            i=j
            lo_h = mid+1
        else:
            hi_h = mid

    if lo_w<lo_hi:
        return np.array([2*lo_r+1, 2*lo_w+1])
    return np.array([2*lo_h+1, 2*lo_r+1])
