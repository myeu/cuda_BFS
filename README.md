cuda_BFS
========

Implements breadth first search for GPU parallel processing using CUDA.

Input file must be in binary and should have the following format (with no newlines):
Number_of_Nodes
Cumulative_count_of_degrees
Adjacency_list

The number of nodes is a single int. The cumulative count of degrees is Number_of_Nodes integers long. The adjacency_list has as many integers as there are edges.

The algorithm runs in iterations, one iteration for every level from the starting node. 
