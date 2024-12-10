import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import networkx as nx
import argparse

def select_initial_seed_set(node_embeddings, second_order_neighbors, k, beta):
    # Initialize the seed set and variables to track influence
    seed_set = set()
    no_embeddings = set()
    node_influence = defaultdict(set)  # Tracks which nodes each node influences

    # Filter second-order neighbors by similarity > beta and track influence
    for node, neighbors in second_order_neighbors.items():
        if node not in node_embeddings:
            no_embeddings.add(node)
            continue

        filtered_neighbors = set()
        for neighbor in neighbors:
            if neighbor not in node_embeddings:
                no_embeddings.add(neighbor)
                continue

            similarity = cosine_similarity(node_embeddings[node].reshape(1, -1), node_embeddings[neighbor].reshape(1, -1))[0][0]
            if similarity > beta:
                filtered_neighbors.add(neighbor)

        node_influence[node] = filtered_neighbors

    # Iteratively select the top-k influential nodes
    influenced_nodes = set()
    while len(seed_set) < k:
        # Select the node with the maximum number of unique influences
        best_node = None
        max_influence = 0

        for node, influences in node_influence.items():
            # Exclude already influenced nodes
            unique_influences = influences - influenced_nodes
            if len(unique_influences) > max_influence:
                best_node = node
                max_influence = len(unique_influences)

        if best_node is None:  # In case no more nodes can be selected
            break

        # Add the best node to the seed set
        seed_set.add(best_node)
        # Update the set of influenced nodes
        influenced_nodes.update(node_influence[best_node])

        # Remove the influenced nodes from all other nodes' influence lists
        for node in node_influence:
            node_influence[node] -= influenced_nodes

    # print(f"Nodes with missing embeddings: {no_embeddings}")
    return list(seed_set)


def get_second_order_neighbors(edge_index):
    # Create a directed graph using NetworkX
    G = nx.DiGraph()
    G.add_edges_from(edge_index)
    
    # Initialize a dictionary to store second-order neighbors
    second_order_neighbors = defaultdict(set)
    
    # For each node, find second-order neighbors
    for node in G.nodes():
        first_order = set(G.successors(node))  # First-order neighbors
        second_order = set()
        
        for neighbor in first_order:
            second_order.update(G.successors(neighbor))  # Second-order neighbors
            
        # Remove the original node from its own second-order neighbors, if present
        second_order.discard(node)
        
        second_order_neighbors[node] = second_order
    
    return second_order_neighbors

def get_first_and_second_hop_neighbors(edge_index, nodes):
    G = nx.DiGraph()
    G.add_edges_from(edge_index)

    first_hop_neighbors = defaultdict(set)
    second_hop_neighbors = defaultdict(set)

    for node in nodes:
        if node in G:
            first_order = set(G.successors(node))
            second_order = set()

            for neighbor in first_order:
                second_order.update(G.successors(neighbor))

            first_hop_neighbors[node] = first_order
            second_hop_neighbors[node] = second_order
            second_hop_neighbors[node].discard(node)

    return first_hop_neighbors, second_hop_neighbors

# Step 2: Filter neighbors by cosine similarity
def filter_neighbors_by_similarity(node_embeddings, neighbors, beta):
    filtered_neighbors = defaultdict(set)
    no_embeddings = set()

    for node, neighbors_set in neighbors.items():
        if node not in node_embeddings:
            no_embeddings.add(node)
            continue

        for neighbor in neighbors_set:
            if neighbor not in node_embeddings:
                no_embeddings.add(neighbor)
                continue

            similarity = cosine_similarity(node_embeddings[node].reshape(1, -1), 
                                           node_embeddings[neighbor].reshape(1, -1))[0][0]

            if similarity > beta:
                filtered_neighbors[node].add(neighbor)

    return filtered_neighbors, no_embeddings

# Step 4: Build MultiDiGraph using filtered neighbors and edge attributes from "Graph1"
def build_multidigraph(graph_data, filtered_first_hop, filtered_second_hop):
    G = nx.MultiDiGraph()

    for node, first_hop_set in filtered_first_hop.items():
        for first_hop in first_hop_set:
            # Add edge between node and first-hop with attributes
            if node in graph_data and first_hop in graph_data[node]:
                attributes_list = graph_data[node][first_hop]
                for attr in attributes_list:
                    topic, weight = attr[0], attr[1]
                    G.add_edge(node, first_hop, topic=topic, weight=weight)

            # Add second-hop neighbors from the original node
            if node in filtered_second_hop:
                for second_hop in filtered_second_hop[node]:
                    if first_hop in graph_data and second_hop in graph_data[first_hop]:
                        attributes_list = graph_data[first_hop][second_hop]
                        for attr in attributes_list:
                            topic, weight = attr[0], attr[1]
                            G.add_edge(first_hop, second_hop, topic=topic, weight=weight)

    return G

# Step 5: Build MultiDiGraph from "Graph1" pickle file
def load_graph_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as file:
        graph_data = pickle.load(file)
    
    return graph_data

# Step 6: Traverse and collect attribute frequencies with single attribute handling and tuple breakdown
def traverse_graph_and_collect_attributes(multidigraph, start_nodes, x):
    """
    Traverse the neighbors of all user-provided nodes (first-hop and second-hop),
    categorize them based on edge attributes, and prioritize based on node counts.
    If the same attribute is encountered in both first-hop and second-hop, treat it as a single attribute.
    Break collective attributes like (topic1, topic2) into individual attributes.
    Store results in a dictionary where the key is the attribute and the value is the length of nodes.
    """
    # Default dictionary to store attribute -> set of destination nodes
    attribute_to_nodes = defaultdict(set)

    # Step 1: Traverse first-hop neighbors
    for node in start_nodes:
        for first_hop, edge_data in multidigraph[node].items():
            for edge_key, edge_attributes in edge_data.items():
                attribute = edge_attributes.get("topic")  # Get the attribute of the edge
                if attribute:
                    attribute_to_nodes[attribute].add(first_hop)  # Add first-hop neighbor to the set
    
        # Step 2: Traverse second-hop neighbors
        for first_hop in multidigraph.successors(node):
            for second_hop, edge_data in multidigraph[first_hop].items():
                for edge_key, edge_attributes in edge_data.items():
                    second_hop_attribute = edge_attributes.get("topic")  # Get the second-hop edge attribute
                    if second_hop_attribute:
                        # Check if the first-hop and second-hop attributes are the same
                        if attribute == second_hop_attribute:
                            # If they are the same, treat as a single attribute
                            attribute_to_nodes[attribute].add(second_hop)
                        else:
                            # Otherwise, consider the combined attributes
                            combined_attributes = (attribute, second_hop_attribute)
                            attribute_to_nodes[combined_attributes].add(second_hop)

    # Step 3: Find the attribute with the most nodes and break down combined topics
    prioritized_attribute_lengths = {}  # Dictionary to store attribute -> length of node set

    while len(prioritized_attribute_lengths) < x and attribute_to_nodes:
        # Sort attributes by the number of nodes in descending order
        most_frequent_attribute = max(attribute_to_nodes, key=lambda k: len(attribute_to_nodes[k]))

        # Handle tuples: break them down into separate attributes
        if isinstance(most_frequent_attribute, tuple):
            for single_topic in most_frequent_attribute:
                if single_topic not in prioritized_attribute_lengths:
                    prioritized_attribute_lengths[single_topic] = len(attribute_to_nodes[most_frequent_attribute])
        else:
            prioritized_attribute_lengths[most_frequent_attribute] = len(attribute_to_nodes[most_frequent_attribute])

        # Remove nodes associated with the most frequent attribute from all other attribute sets
        nodes_to_remove = attribute_to_nodes[most_frequent_attribute]
        del attribute_to_nodes[most_frequent_attribute]

        for attribute in attribute_to_nodes:
            attribute_to_nodes[attribute] -= nodes_to_remove

    return prioritized_attribute_lengths

# Step 7: Get the node IDs from the provided set that influence the top X topics
def get_nodes_influencing_top_topics(multidigraph, top_attributes, provided_nodes):
    """
    For each top attribute, get the nodes from the provided node IDs that are influencing it.
    """
    attribute_to_influencing_nodes = defaultdict(set)

    # Traverse the graph and collect the provided nodes influencing each top attribute
    for u, v, data in multidigraph.edges(data=True):
        attribute = data.get("topic")  # Get the topic of the edge
        if attribute in top_attributes and u in provided_nodes:
            attribute_to_influencing_nodes[attribute].add(u)  # Add the provided node as the influencer

    return attribute_to_influencing_nodes



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Node influence and attribute extraction.")
    parser.add_argument("--beta", type=float, default=0.1, help="Similarity threshold for filtering")
    parser.add_argument("--k", type=int, default=10, help="Number of seed nodes to select.")
    parser.add_argument("--x", type=int, default=5, help="Top attributes to collect.")

    args = parser.parse_args()

    # Load embeddings and edge index
    with open('node_features.pkl', 'rb') as file:
        features = pickle.load(file)
    with open('original_edge_index.pkl', 'rb') as file:
        edge_index = pickle.load(file)

    # Get second-order neighbors
    second_order_neighbors = get_second_order_neighbors(edge_index)

    # Select seed nodes using the values from command line arguments
    seed_nodes = select_initial_seed_set(node_embeddings=features, second_order_neighbors=second_order_neighbors, k=args.k, beta=args.beta)
    print("Selected Seed Nodes:", seed_nodes)

    # Analyze graph
    first_hop, second_hop = get_first_and_second_hop_neighbors(edge_index, seed_nodes)
    filtered_first, _ = filter_neighbors_by_similarity(features, first_hop, args.beta)
    filtered_second, _ = filter_neighbors_by_similarity(features, second_hop, args.beta)

    # Use seed_nodes to get attributes
    graph_data = load_graph_from_pickle('Graph1')
    multidigraph = build_multidigraph(graph_data, filtered_first, filtered_second)
    
    # Pass seed_nodes as the input for attribute traversal
    attributes = traverse_graph_and_collect_attributes(multidigraph, seed_nodes, args.x)
    # print("Top Attributes:", attributes)

    # Step 7: Get the nodes from the provided set that influence the top x attributes
    provided_nodes = set(seed_nodes)
    nodes_influencing_topics = get_nodes_influencing_top_topics(multidigraph, attributes.keys(), provided_nodes)
    
    # Output the result
    print("Top attributes with their node set lengths:", attributes)
    print("Nodes influencing the top attributes:")
    for attribute, nodes in nodes_influencing_topics.items():
        print(f"{attribute}: {', '.join(map(str, nodes))}")