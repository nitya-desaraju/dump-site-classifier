from plotting.py import plot_graph
from create_node_list.py import Node

import networkx as nx
import random
from collections import Counter
import matplotlib.pyplot as plot
import numpy as np

adj = None
nodes = None

def propagate_label(node_idx):
    global adj
    global nodes
    weights = np.zeros(len(nodes))

    for i, n in enumerate(nodes):
        weights[n.label] += adj[node_idx, i]

    new_label = np.argsort(weights)[-1]

    if new_label==nodes[node_idx].label:
        return True
    nodes[node_idx].label = new_label
    return False

def save_current_graph(iteration_num):
    global adj
    global nodes
    fig, ax = plot_graph(nodes, adj)
    plot.savefig(f"GraphImages/Graph{iteration_num}.png", format="PNG")
    return

def whispers(node_list, adj_matrix, num_iterations=50):
    global nodes
    global adj
    nodes = node_list
    adj = adj_matrix

    for i in range(num_iterations):
        curr_node = random.randint(0,len(nodes)-1)
        propagate_label(curr_node)
        save_current_graph(i)
    return
