#ifndef _KERNEL_H
#define _KERNEL_H

//Node contains its offset into packed edge array, and its degree
typedef struct Node {
	int starting;
	int no_of_edges;
}Node;


/*
 *   d_graph_nodes   info about all the nodes, is num. nodes long
 *   d_edge_list     packed edge list, is num. edges long
 *   d_graph_visited true if cost has been set, don't set again 
 *                       later, is num. nodes long
 *   d_cost          distance from start node, is num. nodes long
 *   level           current iteration level
 *   loop            true if a node was found on the frontier
 *   no_of_nodes     number of nodes
 *
 *   >1 thread may try to set cost at the same time,
 *      but its a benign race. They all set it to the same
 *      thing. Only nodes at the same level can set cost.
 */
 
__global__ void
bfs_kernel(Node* d_graph_nodes, int* d_edge_list, bool* d_graph_visited, 
    int* d_cost, int level, bool* loop, int no_of_nodes) 
{  
    // Calculate node id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread only continues if the node is on current level
    if (tid < no_of_nodes && d_cost[tid] == level) {

        // Check neighbors for not-yet-visited nodes
        //      if one is found, set its cost and put it on 
        //      the frontier
        for (int i = d_graph_nodes[tid].starting; i <
                (d_graph_nodes[tid].no_of_edges +
                 d_graph_nodes[tid].starting); i++) 
        {
            int id = d_edge_list[i];

            if (!d_graph_visited[id]) 
            {
                //distance is set to level the vertex is visited
                d_cost[id] = d_cost[tid] + 1;
                
                // cost is set, don't set again
                d_graph_visited[id] = true; 
                
                // a frontier node was found, iterate level again
                *loop = true;
            }
        }
    }
}


#endif