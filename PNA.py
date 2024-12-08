# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 08:17:48 2024

@author: Arpan Dam
"""

import pickle 
import torch
import random
from torch_geometric.data import Data
import torch.optim as optim
#from torch_geometric.nn import GCNConv
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv
#import input_graph_creation
import torch.nn.functional as F
import numpy as np
#import input_small_graph_creation
from pytorch_metric_learning.losses import NTXentLoss

import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import graph_augmentation_v2
import gc
import psutil
import PNA_indegree_sample
from torch_geometric.nn import GCNConv

# Print memory usage
print(psutil.virtual_memory())
#import pyg_lib
'''
import pyg_lib
print(pyg_lib.__version__)
'''
print(torch.__version__)
print(torch_geometric.__version__)
# 1st graph
#input_graph1,node_features,edge_index=input_small_graph_creation.return_input_graph1()


#input_graph10000,node_features,edge_index0000,input_graph20000,edge_index20000=input_graph_creation.return_two_graph()

# 2nd graph
#input_graph2,node_features,edge_index2=input_graph_creation.return_input_graph2()

# Randmly select 2 integer 

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,perceptron_hidden_dim,degree,edge_dim):
        super().__init__()
        self.conv1 = PNAConv(in_channels, hidden_channels,["sum","mean","min","max","var","std"],["identity","amplification","attenuation","linear","inverse_linear"],degree,edge_dim)
        self.conv2 = PNAConv(hidden_channels, out_channels,["sum","mean","min","max","var","std"],["identity","amplification","attenuation","linear","inverse_linear"],degree,edge_dim)
        #self.conv1 = GCNConv(in_channels, hidden_channels)
        #self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc1 = torch.nn.Linear(out_channels, perceptron_hidden_dim)
        self.fc2 = torch.nn.Linear(perceptron_hidden_dim, out_channels)
        
    def forward(self,data1st_graph,data2nd_graph):
        
        # Get the 2 input
        
        #optimizer.zero_grad()
        #input_graph1,node_features,edge_index=input_graph_creation.return_input_graph1()
        #input_graph1,node_features,edge_index=input_small_graph_creation.return_input_graph1()

        # 2nd graph
        #input_graph2,node_features,edge_index2=input_graph_creation.return_input_graph2()
        #input_graph2,node_features,edge_index2=input_small_graph_creation.return_input_graph2()
        # Get representation for the 1st graph
        x, edge_index,edge_attr = data1st_graph.x, data1st_graph.edge_index,data1st_graph.edge_attr
        #################
        '''number_of_edges=len(data1st_graph.edge_index[1])
        #degree=[number_of_edges/2,number_of_edges/2]
        degree=torch.tensor([number_of_edges/2,number_of_edges/2,0])'''
        #edge_features=edge_feature_finder(data1st_graph)
        
        ###################
        x = self.conv1(x, edge_index,edge_attr) 
        x = torch.relu(x)
        x = self.conv2(x, edge_index,edge_attr) 
        
        x = F.relu(self.fc1(x))
        x_graph1 = self.fc2(x)
        #print("")
        #return x
       
        # Get representation for the 2nd graph
        x, edge_index,edge_attr = data2nd_graph.x, data2nd_graph.edge_index,data2nd_graph.edge_attr
        x = self.conv1(x, edge_index,edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index,edge_attr)
        
        x = F.relu(self.fc1(x))
        x_graph2 = self.fc2(x)
        return x_graph1,x_graph2
loss_func = NTXentLoss(temperature=0.10)

'''
#in_channels = node_features.size(1)  # Number of features per node
in_channels=100
hidden_channels = 16  # Example value, you can change it
out_channels = 8  # Example value, you can change it
perceptron_hidden_dim = 32 

model = Net(in_channels, hidden_channels, out_channels,perceptron_hidden_dim)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)'''
def train(train_loader,train_loader2):
    
    model.train()
    loss_list=[]
    for (data1st_graph, data2nd_graph) in zip(train_loader, train_loader2):
        
        
        optimizer.zero_grad()
        #model.train()
        #total_loss = 0
        h_1, h_2= model(data1st_graph,data2nd_graph)
        
        # Prepare for loss
        if(h_2.shape[0]>h_1.shape[0]):
            new_shape=h_1.shape[0]
            h6=h_2[:new_shape]
            h_2=h6
        else :
            new_shape=h_2.shape[0]
            h6=h_1[:new_shape]
            h_1=h6
        embeddings = torch.cat((h_1, h_2))
        # The same index corresponds to a positive pair
        indices = torch.arange(0, h_1.size(0), device=h_2.device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        float_loss=float(loss)
        loss_list.append(float_loss)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
    return sum(loss_list)/len(loss_list)
    #print("")
#input_graph1,node_features,edge_index=input_graph_creation.return_input_graph1()
#input_graph2,node_features,edge_index2=input_graph_creation.return_input_graph2()
#loader=NeighborLoader(input_graph1, num_neighbors=[20, 20], batch_size=128)
'''
for i, s in enumerate(loader):
  print(f'Subgraph: {i:02d}, feature matrix: {s.x.shape}, edges list: {s.edge_index.shape}')'''




def loader_func(input_graph1,input_graph2):
# data in input_graph1
    
    print(input_graph1)
    print(input_graph2)
    #NUM_GRAPHS_PER_BATCH = 64
    #loader = DataLoader(input_graph1, 
                    #batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
    #test_loader = DataLoader(data[int(data_size * 0.8):], 
                         #batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

    #for batch in loader:
    #    print(batch)
    cluster_data = ClusterData(input_graph1, num_parts=128)  # input_graph1 is the data
    #degree=PNA_indegree_sample.compute_degree_centrality_for_normalisation(input_graph1)
    print(psutil.virtual_memory())
    train_loader = ClusterLoader(cluster_data, batch_size=16)
    gc.collect()
    print(psutil.virtual_memory())
    # data in input_graph2
    cluster_data2 = ClusterData(input_graph2, num_parts=128)  # input_graph1 is the data
    print(psutil.virtual_memory())
    train_loader2 = ClusterLoader(cluster_data2, batch_size=16)
    print(psutil.virtual_memory())
    gc.collect()
    #total_num_nodes = 0
    '''
    for step, sub_data in enumerate(train_loader):
      print(f'Batch: {step + 1} has {sub_data.num_nodes} nodes')
      print(sub_data)
      print()
      total_num_nodes += sub_data.num_nodes

    print(f'Iterated over {total_num_nodes} of {input_graph1.num_nodes} nodes!')'''
    return train_loader,train_loader2


def train(train_loader,train_loader2):
    
    model.train()
    loss_list=[]
    for (data1st_graph, data2nd_graph) in zip(train_loader, train_loader2):   # here data1st_graph is the subgraph
        
        #degree1=PNAConv.get_degree_histogram(train_loader)
        #degree2=PNAConv.get_degree_histogram(train_loader2)
        optimizer.zero_grad()
        #model.train()
        #total_loss = 0
        h_1, h_2= model(data1st_graph,data2nd_graph)
        
        # Prepare for loss
        if(h_2.shape[0]>h_1.shape[0]):
            new_shape=h_1.shape[0]
            h6=h_2[:new_shape]
            h_2=h6
        else :
            new_shape=h_2.shape[0]
            h6=h_1[:new_shape]
            h_1=h6
        embeddings = torch.cat((h_1, h_2))
        # The same index corresponds to a positive pair
        indices = torch.arange(0, h_1.size(0), device=h_2.device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        float_loss=float(loss)
        loss_list.append(float_loss)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
    return sum(loss_list)/len(loss_list)
    #print("")

in_channels=100
hidden_channels = 80  # Example value, you can change it
#out_channels = 120  # Example value, you can change it
out_channels = 80 
#perceptron_hidden_dim = 250 
perceptron_hidden_dim = 80 
degree=torch.tensor([240, 328,  79,  39,  23,  12,  11,   7,   6,   5,   7,   3,   1,   0,
          2,   0,   0,   0,   1])

edge_dim=100
model = Net(in_channels, hidden_channels, out_channels,perceptron_hidden_dim,degree,edge_dim)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


############

graph_list=[]
times=1
while(times<3):
    random_choice = np.random.choice([1, 2])
    data_graph=graph_augmentation_v2.func_graph_augmentation(random_choice)  # data_graph not coming
    graph_list.append(data_graph)
    times=times+1
train_loader,train_loader2=loader_func(graph_list[0],graph_list[1])
for epoch in range(1, 1000):
    
    loss = train(train_loader,train_loader2)
    print(loss)
    
#torch.save(model.state_dict(), 'my_model.pth')    
'''
for epoch in range(1, 100):
    # Here randmly select 2 integers
    graph_list=[]
    times=1
    while(times<3):
        random_choice = np.random.choice([1, 2])
        data_graph=graph_augmentation_v2.func_graph_augmentation(random_choice)  # data_graph not coming
        graph_list.append(data_graph)
        times=times+1
    #print(graph_list)
    train_loader,train_loader2=loader_func(graph_list[0],graph_list[1])
    loss = train(train_loader,train_loader2)
    print(loss)
'''   
'''   
in_channels = node_features.size(1)  # Number of features per node
hidden_channels = 16  # Example value, you can change it
out_channels = 8  # Example value, you can change it
perceptron_hidden_dim = 32   # Example value for the perceptron hidden dimension
model = Net(in_channels, hidden_channels, out_channels,perceptron_hidden_dim)'''

# Pass the data through the network
'''
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient computation
    out = model(input_graph1)'''

# Print the output from conv2
#print(out)    
print("")