import numpy as np
import pickle
import requests
from PIL import Image
from io import BytesIO

def id_to_url(target_id, pkl_path="image_ids_filled.pkl"):
    with open(pkl_path, "rb") as f:
        image_objs = pickle.load(f)

    for image in image_objs:
        if int(image.name) == int(target_id):
            return image.url
    
    return None

def get_similarity_score(W_query_norm, pklurl='image_ids_filled.pkl', k=5):

    with open(pklurl, 'rb') as f:
        image_embeddings = pickle.load(f)

    image_ids = []
    semantic_vectors = []

    for image in image_embeddings:
        vec = image.descriptor
        vec = vec / np.linalg.norm(vec)  #normalize
        image_ids.append(image.name)
        semantic_vectors.append(vec)

    semantic_matrix = np.stack(semantic_vectors)  #shape (N, 50) (W_norm should be (1, 50))

    similarities = np.dot(semantic_matrix, W_query_norm.T).flatten()

    
    

    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_ids = [image_ids[i] for i in top_k_indices]
    top_urls = [id_to_url[int(img_id)] for img_id in top_k_ids]

    return top_urls

def get_image(top_urls):
    images = []
    for url in top_urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        images.append(img)
    return images
