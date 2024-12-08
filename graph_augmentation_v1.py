# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:58:16 2024

@author: Arpan Dam
"""

import pickle 
import torch
import random
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import math
from math import log
#file_path = 'edge_probability_career'

file_path = 'edge_probability_career'


eigen_vector='eigenvector.pkl'

pagerank='page_rank.pkl'

outdegree='outdegree.pkl'

indegree='indegree.pkl'

node_to_index_mapping='node_to_index_mapping.pkl'

with open(file_path, 'rb') as file:
    # Load the data from the file
    edge_probability_career = pickle.load(file)
    
    
with open(eigen_vector, 'rb') as file:
    # Load the data from the file
    eigenvector = pickle.load(file)
    
with open(pagerank, 'rb') as file:
    # Load the data from the file
    pagerank = pickle.load(file)
        

with open(outdegree, 'rb') as file:
    # Load the data from the file
    outdegree = pickle.load(file)
    
with open(indegree, 'rb') as file:
    # Load the data from the file
    indegree = pickle.load(file)
    
    
with open(node_to_index_mapping, 'rb') as file:
    # Load the data from the file
    node_to_index_mapping = pickle.load(file)    
        
print("")



#########################################################################################################################
# augmenting with outdegee

def outdegree_graph1():
    list_outdegree=[]
    for key in outdegree:
        list_outdegree.append(outdegree[key])
        
    #outdegree_average=sum(list_outdegree)/len(list_outdegree) 
    max_outdegree=max(list_outdegree)
    min_outdegree=min(list_outdegree)
    
    #print("")  
        
    dict1_out_degree={}  # this file will be like edge_probablity career having probaboility as outdegree
    
    list_follower=set()
    for key in edge_probability_career:
        dict1={}
        for follower in edge_probability_career[key]:
            list1=[]
            for edge in edge_probability_career[key][follower]:
                '''
                if outdegree[follower]==0:
                    list_follower.add(follower)
                    probability =0
                else: '''   
                    #probability=float((log(outdegree[follower])-0)/(log(max_outdegree)-0)) + ()
                source_node_degree=outdegree[key]
                tail_node_degree=outdegree[follower]
                probability=float(source_node_degree+tail_node_degree)/2
                #probability=float(((outdegree[follower])-(min_outdegree))/((max_outdegree)-(min_outdegree)))
                #probability=0.4
                #print("")
                tag=edge[0]
                list1.append((tag,probability))
            dict1[follower]=list1
        dict1_out_degree[key]=dict1
    return dict1_out_degree
    
print("")    
            
            
#########################################################################################################################

# augmenting with indegree

def indegree_graph1():
    list_indegree=[]
    for key in indegree:
        list_indegree.append(indegree[key])
        
    #indegree_average=sum(list_indegree)/len(list_indegree) 
    max_indegree=max(list_indegree)
    min_indegree=min(list_indegree)
    #print("")  
        
    dict1_in_degree={}  # this file will be like edge_probablity career having probaboility as outdegree
    
    
    for key in edge_probability_career:
        dict1={}
        for follower in edge_probability_career[key]:
            list1=[]
            for edge in edge_probability_career[key][follower]:
                probability=float(indegree[follower]-min_indegree)/(max_indegree-min_indegree)
                #probability=0.4
                #print("")
                tag=edge[0]
                list1.append((tag,probability))
            dict1[follower]=list1
        dict1_in_degree[key]=dict1  
    return dict1_in_degree 
    
#print("")    
            
            
            
#########################################################################################################################
 
    
# augmenting with pagerank

def pagerank1():
    list_pagerank=[]
    for key in pagerank:
        list_pagerank.append(pagerank[key])
        
    #indegree_average=sum(list_indegree)/len(list_indegree) 
    max_pagerank=max(list_pagerank)
    min_pagerank=min(list_pagerank)
    #print("")  
        
    dict1_pagerank={}  # this file will be like edge_probablity career having probaboility as outdegree
    
    
    for key in edge_probability_career:
        dict1={}
        for follower in edge_probability_career[key]:
            list1=[]
            for edge in edge_probability_career[key][follower]:
                source_node_degree=pagerank[key]
                tail_node_degree=pagerank[follower]
                probability=float(source_node_degree+tail_node_degree)/2
                #probability=0.4
                #print("")
                tag=edge[0]
                list1.append((tag,probability))
            dict1[follower]=list1
        dict1_pagerank[key]=dict1  
    return dict1_pagerank
        
#print("")  

#########################################################################################


# augmenting with eigenvector

def eigenvector1():
    list_eigenvector=[]
    for key in eigenvector:
        list_eigenvector.append(eigenvector[key])
        
    #indegree_average=sum(list_indegree)/len(list_indegree) 
    max_eigenvector=max(list_eigenvector)
    min_eigenvector=min(list_eigenvector)
    #print("")  
        
    dict1_eigenvector={}  # this file will be like edge_probablity career having probaboility as outdegree
    
    
    for key in edge_probability_career:
        dict1={}
        for follower in edge_probability_career[key]:
            list1=[]
            for edge in edge_probability_career[key][follower]:
                source_node_degree=eigenvector[key]
                tail_node_degree=eigenvector[follower]
                probability=float(source_node_degree+tail_node_degree)/2
                #probability=float(eigenvector[follower]-min_eigenvector)/(max_eigenvector-min_eigenvector)
                #probability=0.4
                #print("")
                tag=edge[0]
                list1.append((tag,probability))
            dict1[follower]=list1
        dict1_eigenvector[key]=dict1
    return dict1_eigenvector
    
print("")  