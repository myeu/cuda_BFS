#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>

using namespace std;

#include "bfs_kernel.cu"

/*
 * gpu_bfs.cu
 *
 *	Usage: ./executable <graph_file> <output file>
 *
 *	Input: Name of the file containing the graph. Expected format
 *			is binary with the following data (with no newlines)
 *			number of nodes (4 bytes)
 *			cumulative list of node degrees (4 bytes * numer of nodes)
 *			edge list (4 bytes * number of links)
 *			
 *			example: 4 nodes; node 0 edges: 1,2; node 1 edges: 0; node 1 edges: 3
 *			4
 *			0,2,3
 *			1,2,0,3
 *			written in binary with no newlines or commas
 *
 *	Output:
 *		stdout: Time it took kernel to run, in microseconds
 *		output file: Each node and its distance from starting node
 *					(0) cost:0
 *					(1) cost:3
 *		
 */

const int MAX_THREADS_PER_BLOCK = 256;
char *infile = NULL;

int starting_node_id;

int nb_nodes;
int nb_links;
int *degrees;
int *starting;
int *links;


void readFile(char *filename)
{
	ifstream finput(filename, ios::in | ios::binary);
	if(!finput.is_open())
	{
		cout << "Unable to open file" << endl;
		exit(EXIT_FAILURE);
	}

	// Read number of nodes, first 4 bytes of file
	finput.read((char*)&nb_nodes, 4);

	if (starting_node_id < 0 || starting_node_id > nb_nodes)
	{
		cerr << "Starting position is invalid" << endl;
		exit(EXIT_FAILURE);
	}

	// Read cumulative degrees, 4 bytes per node
	degrees = new int[nb_nodes];
	finput.read((char*) degrees, nb_nodes * 4);
	
	starting = new int[nb_nodes];
	memset(starting, 0, sizeof(int) * nb_nodes);
	for (int i = 1; i < nb_nodes; i++)
	{
		starting[i] = degrees[i - 1];
	}

	// Read links, 4 bytes per link
	nb_links = degrees[nb_nodes - 1];
	links = new int[nb_links];
	finput.read((char*) links, nb_links * 4);

	finput.close();
}


void bfsGraph(char *outFile)
{
	// allocate host memory
	Node *h_graph_nodes = (Node *) malloc(sizeof(Node) * nb_nodes);
	bool* h_graph_visited = (bool *) malloc(sizeof(bool) * nb_nodes);

	// Initialize memory of nodes
	h_graph_nodes[0].starting = 0;
	h_graph_nodes[0].no_of_edges = degrees[0];
	h_graph_visited[0] = false;
	for (unsigned int i = 1; i < nb_nodes; i++)
	{
		h_graph_nodes[i].starting = starting[i];
		h_graph_nodes[i].no_of_edges = degrees[i] - degrees[i-1];
		h_graph_visited[i] = false;
	}
	h_graph_visited[starting_node_id] = true;

	// Copy node list to cuda memory
	Node *d_graph_nodes;
	cudaMalloc((void **) &d_graph_nodes, sizeof(Node) * nb_nodes);
	cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) *
		nb_nodes, cudaMemcpyHostToDevice);

	// Copy edge list to device memory
	int *d_edge_list;
	cudaMalloc((void **) &d_edge_list, sizeof(int) * nb_links);
	cudaMemcpy(d_edge_list, links, sizeof(int) * nb_links,
		cudaMemcpyHostToDevice);

	// Copy the visted array to device memory
	bool *d_graph_visited;
	cudaMalloc((void **) &d_graph_visited, sizeof(bool) * nb_nodes);
	cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(bool) *
		nb_nodes, cudaMemcpyHostToDevice);

	// Allocate memory for the result on host
	int *h_cost = (int *) malloc(sizeof(int) * nb_nodes);
	for (int i = 0; i < nb_nodes; i++)
	{
		h_cost[i] = -1;
	}
	h_cost[starting_node_id] = 0;

	// Allocate device memory for result
	int *d_cost;
	cudaMalloc((void **) &d_cost, sizeof(int) * nb_nodes);
	cudaMemcpy(d_cost, h_cost, sizeof(int) * nb_nodes,
		cudaMemcpyHostToDevice);

	// Determine number of blocks and threads
	int num_of_blocks = 1;	// at least 1
	int num_of_threads_per_block = nb_nodes;
	if (nb_nodes > MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int) ceil((double) nb_nodes/
			(double) MAX_THREADS_PER_BLOCK);
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
	}

	bool *d_over;
	cudaMalloc((void **) &d_over, sizeof(bool));
	bool stop;
	int level = 0;

	// Call the kernel at each level iteration
	struct timeval start, end;    
	gettimeofday(&start, NULL);
	do 
	{
		stop = false;
		cudaMemcpy(d_over, &stop, sizeof(bool), 
			cudaMemcpyHostToDevice);

		bfs_kernel<<<num_of_blocks, 
		num_of_threads_per_block>>>(d_graph_nodes, d_edge_list,
			 d_graph_visited, d_cost, level, d_over, nb_nodes);
		
		cudaThreadSynchronize();

		cudaMemcpy(&stop, d_over, sizeof(bool),
			cudaMemcpyDeviceToHost);
		level++;

	} while(stop);

	gettimeofday(&end, NULL);

	// print duration of all iterations of kernel execution
	printf("%ld\n",
           (end.tv_sec * 1000000 + end.tv_usec)
           - (start.tv_sec * 1000000 + start.tv_usec));
	
	cudaMemcpy(h_cost, d_cost, sizeof(int) * nb_nodes,
		cudaMemcpyDeviceToHost);
	
	cudaMemcpy(h_graph_visited, d_graph_visited, sizeof(bool) *
			nb_nodes, cudaMemcpyDeviceToHost);

	// Store results into a file
	FILE *fpo = fopen(outFile, "w");
	for (int i = 0; i < nb_nodes; i++)
	{
		fprintf(fpo, "(%d) cost:%d\n", i, h_cost[i]);
	}
	fclose(fpo);
	
	// clean up memory
	free(h_graph_nodes);
	free(links);
	free(h_graph_visited);
	free(h_cost);
	cudaFree(d_graph_nodes);
	cudaFree(d_edge_list);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);

}

int main(int argc, char **argv)
{
	// Code for printing distances to an output file
	if (argc != 3)
	{
		printf("Usage: %s <input file> <output file>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	char *filename = argv[1];
	char*outFile = argv[2];
	
	readFile(filename);
	
	starting_node_id = rand() % nb_nodes;

	bfsGraph(outFile);
	return 0;
}