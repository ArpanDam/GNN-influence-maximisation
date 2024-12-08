# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:50:07 2024

@author: Arpan Dam
"""

import pickle 
import torch
import random
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import math
from math import log
import graph_augmentation_v1
import random
#file_path = 'edge_probability_career'

file_path = 'edge_probability_career'


eigen_vector='eigenvector.pkl'

pagerank='page_rank.pkl'

outdegree='outdegree.pkl'

indegree='indegree.pkl'


#edge_features='edge_features_v2'

node_to_index_mapping='node_to_index_mapping.pkl'
import numpy as np

node_features="node_features.pkl"


with open('edge_features_v2', 'rb') as file:
    # Load the data from the file
    edge_features = pickle.load(file)

'''
with open('edge_features', 'rb') as file:
    # Load the data from the file
    edge_features = pickle.load(file)'''
#print(random_array)
with open('edge_probability_career', 'rb') as file:
    # Load the data from the file
    edge_probability_career = pickle.load(file)
    
 
with open('node_features.pkl', 'rb') as file:
    # Load the data from the file
    node_features = pickle.load(file)


set_all_nodes=set() # contains ID of all nodes

for key in edge_probability_career:
    for follower in edge_probability_career[key]:
        set_all_nodes.add(key)
        set_all_nodes.add(follower)
        

dict_mapping={} # key node id value index id
index=0

for node in set_all_nodes:
    dict_mapping[node]=index
    index=index+1

        
list_node_features=[] # 0 index node feature will be first

for key in dict_mapping:
    try:
        list_node_features.append(node_features[key])
    except:
        list_node_features.append(np.random.normal(loc=0, scale=1, size=100).astype(np.float32))
 



outdegree_graph=graph_augmentation_v1.outdegree_graph1()

indegree_graph=graph_augmentation_v1.indegree_graph1()

eigenvector_graph=graph_augmentation_v1.eigenvector1()

pagerank_graph=graph_augmentation_v1.pagerank1()


random_augentation_number=2 



def func_graph_augmentation(random_augentation_number):
#random_augentation_number=2

    if(random_augentation_number==1): # do outdegree ougmentation
        #print("") 
        dict_outdegree_edge_removed={}  # this dictionary will store outdegree_graph after remoiving edge randomly
        for key in outdegree_graph:
            dict1={}
            for follower in outdegree_graph[key]: 
                list1=[]
                for edge in outdegree_graph[key][follower]:
                    if(edge[1]*0.4>random.uniform(0, 1)):  # 0.3
                        tag=edge[0]
                        probability=edge[1]
                        list1.append((tag,probability))
                if(len(list1)>0):        
                    dict1[follower]=list1
            if(follower in dict1):        
                dict_outdegree_edge_removed[key]=dict1 
        data_graph=graph_return(dict_outdegree_edge_removed)
        
        return  data_graph                   
                
    '''            
    random_augentation_number=3    
    if(random_augentation_number==3): # do pagerank ougmentation
        print("") 
        dict_pagerank_edge_removed={}  # this dictionary will store outdegree_graph after remoiving edge randomly
        for key in pagerank_graph:
            dict1={}
            for follower in pagerank_graph[key]: 
                list1=[]
                for edge in pagerank_graph[key][follower]:
                    if(edge[1]>random.uniform(0, 1)):
                        tag=edge[0]
                        probability=edge[1]
                        list1.append((tag,probability))
                if(len(list1)>0):        
                    dict1[follower]=list1
            if(follower in dict1):        
                dict_pagerank_edge_removed[key]=dict1[follower] 
    random_augentation_number=4            
    if(random_augentation_number==4): # do eigenvector ougmentation
        print("") 
        dict_eigenvector_edge_removed={}  # this dictionary will store outdegree_graph after remoiving edge randomly
        for key in eigenvector_graph:
            dict1={}
            for follower in eigenvector_graph[key]: 
                list1=[]
                for edge in eigenvector_graph[key][follower]:
                    if(edge[1]>random.uniform(0, 1)):
                        tag=edge[0]
                        probability=edge[1]
                        list1.append((tag,probability))
                if(len(list1)>0):        
                    dict1[follower]=list1
            if(follower in dict1):        
                dict_eigenvector_edge_removed[key]=dict1[follower]  '''
    
    #random_augentation_number=3           
    if(random_augentation_number==2): # do eigenvector ougmentation
        #print("") 
        dict_inf_probablity_removed={}  # this dictionary will store outdegree_graph after remoiving edge randomly
        for key in edge_probability_career:
            dict1={}
            for follower in edge_probability_career[key]: 
                list1=[]
                for edge in edge_probability_career[key][follower]:
                    if(edge[1]*0.8>random.uniform(0, 1)):   # 0.2
                        tag=edge[0]
                        probability=edge[1]
                        list1.append((tag,probability))
                if(len(list1)>0):        
                    dict1[follower]=list1
            if(follower in dict1):        
                dict_inf_probablity_removed[key]=dict1
        data_graph=graph_return(dict_inf_probablity_removed) 
        
        return  data_graph             
# create the graph data here######################################################################################

def graph_return(graph):
    list_source=[]
    list_target=[]
    edge_features_view=[]
    #edge_features_random_graph
    for key in graph:
        for follower in graph[key]:
            list_source.append(dict_mapping[key])
            list_target.append(dict_mapping[follower])
            edge_features_view.append(edge_features[key][follower])
    edge_index = torch.tensor([list_source, list_target], dtype=torch.long)
    node_features = torch.tensor(list_node_features, dtype=torch.float)
    result_tensor = torch.stack(edge_features_view)
    result_tensor = torch.squeeze(result_tensor, dim=1)          # this result tensor is the edge features
    data_graph = Data(x=node_features, edge_index=edge_index,edge_attr=result_tensor) 
    return data_graph
#print("") 
#return data_graph,node_features,edge_index    
#print("")
func_graph_augmentation(random_augentation_number)
'''
list_source_new = list_source[:]
list_target_new = list_target[:]
number_of_edge_to_remove=20
index=0
while(index<number_of_edge_to_remove):
#for i in len(list_source_new):
    unique_random_integers = random.sample(range(len(list_source_new)), 1)
    #list1=[]
    #list1.append(unique_random_integers)
    list_source_new.pop(unique_random_integers[0])
    list_target_new.pop(unique_random_integers[0])
    index=index+1
print("")    


edge_index2 = torch.tensor([list_source_new, list_target_new], dtype=torch.long)     
data_graph2 = Data(x=node_features, edge_index2=edge_index2) '''


