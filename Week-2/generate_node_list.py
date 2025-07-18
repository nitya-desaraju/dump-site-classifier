from generate_adjacency_matrix.py import generate_adjacency_matrix
from create_node_list.py import create_node_list
from cos_dist.py import cos_dist
from whispers.py import whispers

def imgpaths_to_graph(paths: list[str], C: float):
    whispers(generate_node_list(paths, C))

def generate_node_list(paths: list[str], C: float) -> (np.ndarray, list):
    descriptors = img_to_descriptors_array(paths)
    dists = cos_dist(descriptors, descriptors)
    adjs = generate_adjacency_matrix(dists, C)
    nodes = create_node_list(paths, adjs)
    return nodes, adjs
