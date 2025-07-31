import numpy as np
import cv2
import pickle
import PIL.Image

class Image:
    def __init__(self, idd, heat):
        self.id = idd
        self.heat_map = heat

#ONE IMAGE, size-tensor int
def identify_dump(image, size_tensor):
    
    size_tensor = size_tensor[0]

    assert size_tensor % 2 == 1, "size_tensor must be odd "
    
    pad = size_tensor // 2
    padded_img = np.pad(image, pad_width=pad, mode='constant', constant_values=0)

    heatmap_sum = cv2.boxFilter(padded_img, ddepth=-1, ksize=(size_tensor, size_tensor), normalize=False)

    valid_area = heatmap_sum[pad:pad + image.shape[0], pad:pad + image.shape[1]]

    y, x = np.unravel_index(np.argmax(valid_area), valid_area.shape)

    return (x, y)


bitmasks = None # prefix sum array
def maps2masks(heatmaps, threshold=0.25):
    global bitmasks

    bitmasks = np.zeros(np.array(heatmaps.shape)+np.array([0,0,1,1]))
    print("maps2masks")
    
    for i,x in enumerate(heatmaps):
        for j,y in enumerate(x[0]):
            for k,z in enumerate(y):
                if z<=0.25:
                    bitmasks[i,0,j+1,k+1] = 1
                bitmasks[i,0,j+1,k+1] += bitmasks[i,0,j,k+1] + bitmasks[i,0,j+1,k] - bitmasks[i,0,j,k]
    
    bitmasks.reshape((bitmasks.shape[0],bitmasks.shape[2],bitmasks.shape[3]))

    with open("maps2masks.pkl", "wb") as f:
        pickle.dump(bitmasks, f)

    return


def optimal_bounding_box(heatmaps : np.ndarray, fitting=0.0, threshold=0.25):
    '''
    fitting represents the maximum allowed percentage (0.0-1.0) of 
    below-threshold (0.25) heat values within the bounding box
    '''
    
    #maps2masks(heatmaps, threshold)
    global bitmasks
    with open("maps2masks.pkl", "rb") as f:
        bitmasks = pickle.load(f)

    lo_r = 0
    hi_r = heatmaps.shape[2]//4
    
    i=0
    #FINDING THE OPTIMAL SQUARE BONDING BOX
    while lo_r < hi_r:
        mid = (lo_r + hi_r)//2
        
        j=i
        while j<heatmaps.shape[0]:
            x,y = np.array(identify_dump(heatmaps[j,0],(2*mid+1,2*mid+1))) + np.array([1,1])
            print(mid, x,y)
            
            count = bitmasks[j, x+mid, y+mid] - bitmasks[j, x-mid-1, y+mid] - \
                    bitmasks[j, x+mid, y-mid-1] + bitmasks[j, x-mid-1, y-mid-1]
            
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
            x,y = np.array(identify_dump(heatmaps[j,0],(2*lo_r+1,2*mid+1))) + np.array([1,1])
            count = bitmasks[j,x+lo_r,y+mid] - bitmasks[j,x-lo_r-1,y+mid] - bitmasks[j,x+lo_r,y-mid-1] + bitmasks[j,x-lo_r-1,y-mid-1]
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
            x,y = np.array(identify_dump(heatmaps[j,0],(2*mid+1,2*lo_r+1))) + np.array([1,1])
            count = bitmasks[j,x+mid,y+lo_r] - bitmasks[j,x-mid-1,y+lo_r] - bitmasks[j,x+mid,y-lo_r-1] + bitmasks[j,x-mid-1,y-lo_r-1]
            if not (count/((2*mid+1)*(2*lo_r+1))**2<=fitting):
                break
            j+=1
        if j<heatmaps.shape[0]:
            i=j
            lo_h = mid+1
        else:
            hi_h = mid

    if lo_w<lo_h:
        return np.array([2*lo_r+1, 2*lo_w+1])
    return np.array([2*lo_h+1, 2*lo_r+1])


def resize_heatmaps():
    with open("image_objects_partial.pkl", "rb") as f:
        images = pickle.load(f)

    heatmaps = [image.heat_map for image in images]

    size = (500, 500)
    resized_heatmaps = []

    for heatmap in heatmaps:
        if heatmap.ndim == 3 and heatmap.shape[0] == 1:
            heatmap = heatmap[0]

        norm = (255 * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)).astype(np.uint8)
        
        img = PIL.Image.fromarray(norm)

        resized = img.resize(size, resample = PIL.Image.BILINEAR)
        resized_np = np.array(resized, dtype = np.float32) / 255.0

        resized_np = np.expand_dims(resized_np, axis = 0)
        resized_heatmaps.append(resized_np)

    resized_array = np.stack(resized_heatmaps)

    with open("resized_heatmaps.pkl", "wb") as f:
        pickle.dump(resized_array, f)



with open("resized_heatmaps.pkl", "rb") as f:
    heatmaps = pickle.load(f)

print(np.array(heatmaps).shape)
optimal_shape = optimal_bounding_box(heatmaps)

print(f"Optimal shape: {optimal_shape}")
