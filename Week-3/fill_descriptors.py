import pickle

def fill_descriptors(features_path, image_ids_path="image_ids.pkl"):
    with open(image_ids_path, "rb") as f:
        image_ids = pickle.load(f)

    with open(features_path, "rb") as f:
        resnet18_features = pickle.load(f)

    updated_images = []

    for image in image_ids:
        original_name = image.name
        formatted_name = f"COCO_val2014_{int(original_name):012}.jpg"

        if formatted_name in resnet18_features:
            image.descriptor = resnet18_features[formatted_name]
            updated_images.append(image)
        else:
            continue

    with open("image_ids_updated.pkl", "wb") as f:
        pickle.dump(updated_images, f)