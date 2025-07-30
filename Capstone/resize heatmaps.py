import pickle
import numpy as np
from PIL import Image

with open("heatmaps.pkl", "rb") as f:
    heatmaps = pickle.load(f)

size = (500, 500)
resized_heatmaps = []

for heatmap in heatmaps:
    if heatmap.ndim == 3 and heatmap.shape[0] == 1:
        heatmap = heatmap[0]

    norm = (255 * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)).astype(np.uint8)
    img = Image.fromarray(norm)

    resized = img.resize(size, resample = Image.BILINEAR)
    resized_np = np.array(resized, dtype = np.float32) / 255.0

    resized_np = np.expand_dims(resized_np, axis = 0)
    resized_heatmaps.append(resized_np)

resized_array = np.stack(resized_heatmaps)

with open("resized_heatmaps.pkl", "wb") as f:
    pickle.dump(resized_array, f)