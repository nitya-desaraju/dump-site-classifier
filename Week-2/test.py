from img_to_descriptors import img_to_descriptors
from predict_face_from_image import predict_face_from_image
import skimage.io as io
from facenet_models import FacenetModel

image = io.imread(r"")
if image.shape[-1] == 4:
    # Image is RGBA, where A is alpha -> transparency
    image = image[..., :-1]  # png -> RGB

try: 
    descriptors, filtered_boxes = img_to_descriptors(image, 0.91)
    predict_face_from_image(descriptors, filtered_boxes, image, 0.5)
except:
    print("No face detected")
