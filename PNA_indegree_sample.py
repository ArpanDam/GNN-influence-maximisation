# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 09:01:55 2024

@author: Arpan Dam
"""

import torch
import networkx as nx
#from torch_geometric.data import Data

# Example in-degrees tensor
d = torch.tensor([0,0, 1, 1, 2, 2, 2, 3, 3, 3])

# Assume the maximum degree is 4
max_degree = 4

# Initialize the histogram tensor
deg = torch.zeros(max_degree + 1, dtype=torch.long)

# Compute the histogram and add to the tensor
deg += torch.bincount(d, minlength=deg.numel())

print(deg)  # Output: tensor([1, 2, 3, 4, 0])


# We get the input graph in this function
def compute_degree_centrality_for_normalisation(input_graph):
    edge_list = input_graph.edge_index.t().tolist()
    
    G = nx.DiGraph(edge_list)
    in_degrees = dict(G.in_degree())
    max_in_degrees = max(in_degrees.values())
    deg = torch.zeros(max_in_degrees + 1, dtype=torch.long)
    values = list(in_degrees.values())
    tensor = torch.tensor(values)
    deg += torch.bincount(tensor, minlength=deg.numel())
    return deg
    print("")

