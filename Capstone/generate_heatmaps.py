from utils.image_processor import ImageProcessor
import os
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image

image_dir = "images0"
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(".png")]

cam_array = []

ip = ImageProcessor(CATS, STATE_DICT_PATH, model=model)

for img_path in tqdm(image_paths):
    try:
        #print("Processing:", img_path)

        iw = ip.execute_cams_pred(img_path)

        if iw.global_cams is None or len(iw.global_cams) == 0:
            print(f"No CAMs returned for {img_path}")
            continue

        cam = iw.global_cams[0]

        cam_resized = np.array(Image.fromarray(cam).resize((800, 800)))
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        cam_resized = np.expand_dims(cam_resized, axis=0)

        cam_array.append(cam_resized)

    except Exception as e:
        print("Error on", img_path, ":", str(e))


if cam_array:
    cam_array = np.stack(cam_array)
    with open("cams_part0_final.pkl", "wb") as f:
        pickle.dump(cam_array, f)
    print("Saved CAM array with shape:", cam_array.shape)
else:
    print("No CAMs were saved.")
