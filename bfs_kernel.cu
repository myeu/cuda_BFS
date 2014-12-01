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
 *   d_graph_level   true if the node is on the frontier (not yet 
 *                       discovered), is num. nodes long
 *   d_graph_visited true if cost has been set, don't set again, 
 *                       is num. nodes long
 *   d_cost          distance from start node, is num. nodes long
 *   loop            true if a node was found on the frontier
 *   no_of_nodes     number of nodes
 */
__global__ void
bfs_kernel(Node* d_graph_nodes, int* d_edge_list, bool* d_graph_level,
        bool* d_graph_visited, int* d_cost, bool* loop, int no_of_nodes) 
{  
    // Calculate node id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread only continues if the node is on the frontier
    if (tid < no_of_nodes && d_graph_level[tid]) {

        // Don't process this node next iteration
        d_graph_level[tid] = false;

        // Check neighbors for not-yet-visited nodes
        //      if one is found, set its cost and put it on 
        //      the frontier
        for (int i = d_graph_nodes[tid].starting; i <
                (d_graph_nodes[tid].no_of_edges +
                 d_graph_nodes[tid].starting); i++) 
        {
            int id = d_edge_list[i];

            if (!d_graph_visited[id]) {
                //calculate in which level the vertex is visited
                d_cost[id] = d_cost[tid] + 1;
                d_graph_level[id] = true;
                d_graph_visited[id] = true; 
                // a frontier node was found, iterate level again
                *loop = true;
            }
        }
    }
}


#endif