import pickle
from cogworks_data.language import get_data_path
from pathlib import Path

class image_id:
    def __init__(self, name, url):
        self.name = name
        self.url= url
        self.caption_ID = []
        self.descriptor= None
        self.W= None

def fill_descriptors(image_ids_path="image_ids.pkl"):
    with open(image_ids_path, "rb") as f:
        image_ids = pickle.load(f)

    with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
        resnet18_features = pickle.load(f)

    updated_images = []

    for image in image_ids:
        name = image.name

        if name in resnet18_features:
            image.descriptor = resnet18_features[name]
            updated_images.append(image)
        else:
            continue

    with open("/Users/veeradesale/Desktop/updated_images/image_ids_updated.pkl", "wb") as f:
        pickle.dump(updated_images, f)

fill_descriptors(image_ids_path="image_ids.pkl")
