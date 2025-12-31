# GNN-influence-maximisation


## This is the implementation of the paper "Graph Contrastive Learning for Tag-Aware Influence Maximization"

Importance score folder contains the code of finding imporance scorte of all the nodes. 

Run : python importent_score.py to generate files stroing importance score of each nodes.


A toy graph Graph1 is provided which is fully anonymized, The toy graph is in the form of dictionary and store in pickle file.

1) Run `python seed_tag.py'  to run the code with k(number of influencial user), r(number of influence tags) and beta as 10 , 5 and 0.1.
2)  Run `python seed_tag.py --beta 0.3 --k 5 --x 4' to run the code with number of influencial user =5, number of influence tags = 4 and beta = 0.3

