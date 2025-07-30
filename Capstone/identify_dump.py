import numpy as np
import cv2

#ONE IMAGE, size-tensor int
def identify_dump(image, size_tensor):
    assert size_tensor % 2 == 1, "size_tensor must be odd "
    
    pad = size_tensor // 2
    padded_img = np.pad(image, pad_width=pad, mode='constant', constant_values=0)

    heatmap_sum = cv2.boxFilter(padded_img, ddepth=-1, ksize=(size_tensor, size_tensor), normalize=False)

    valid_area = heatmap_sum[pad:pad + image.shape[0], pad:pad + image.shape[1]]

    y, x = np.unravel_index(np.argmax(valid_area), valid_area.shape)

    return (x, y)

