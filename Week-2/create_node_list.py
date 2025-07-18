import numpy as np

#Node class for use in create_node_list
class Node:
    def __init__(self, ID, neighbors, truth=None, file_path=None):
        self.id = ID                       #Unique ID
        self.label = ID                    #Initially same as ID (Changed in whispers algorithm)
        self.neighbors = neighbors         #IDs of connected nodes                   
        self.file_path = file_path         

    def __repr__(self):
        return f'Node {self.label} with ID:{self.id} connected to nodes: {self.neighbors}'

    def relabel(self, newname):
        self.label = newname

def create_node_list(paths: list[str], adjs: np.ndarray) -> list[Node]:
    nodes = []
    for i, path in enumerate(paths):
        neighbors = tuple(j for j, weight in enumerate(adjs[i]) if weight > 0 and i != j)
        
        node = Node(
            ID=i,
            neighbors=neighbors,
            file_path=path
        )
        
        nodes.append(node)
    return nodes
