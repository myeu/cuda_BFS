/*
 * main.cpp
 *
 *  Created on: Nov 27, 2014
 *      Author: rtamaskar
 */
#include<iostream>
#include<fstream>
#include<stdio.h>
#include<stdlib.h>
#include<string>
#include <cstdlib>
#include <pthread.h>
#include <cstring>

using namespace std;

# define num_of_threads 10

void bfsGraph(char* filename, int start_position);

void* bfs_parallel(void *n);

typedef struct Node {
	int starting;
	int no_of_edges;
}Node;

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("Usage: %s <input file> \n", argv[0]);
		exit(EXIT_FAILURE);
	}
	char* filename = argv[1];
	bfsGraph(filename, 0);
}

int nb_nodes;
Node* h_graph_nodes;
bool* h_graph_level;
bool* h_graph_visited;
int* degrees;
int* starting;
int* links;
int* h_cost;
bool d_over;
pthread_mutex_t the_mutex;

void bfsGraph(char* filename, int start_position) {
	ifstream finput;
	pthread_mutex_init(&the_mutex, NULL);	
	finput.open(filename, ios::in | ios::binary);

	//Read number of nodes on 4 bytes
	finput.read((char*)&nb_nodes, 4);
	if (start_position < 0 || start_position > nb_nodes) {
		return;
	}
	
	//Read cumulative degrees
	degrees = new int[nb_nodes];
	finput.read((char*)degrees, nb_nodes * 4);

	starting = new int[nb_nodes];
	memset(starting, 0, sizeof(int) * nb_nodes);
	for (int i = 1; i < nb_nodes; i++) {
		starting[i] = degrees[i - 1];
	}

	//Read links
	int nb_links = degrees[nb_nodes - 1];
	links = new int[nb_links];
	finput.read((char*)links, nb_links * 4);

	finput.close();

	cout<<"Number of nodes are "<<nb_nodes<<endl;

	//allocate host memory
	h_graph_nodes = (Node*) malloc(sizeof(Node) * nb_nodes);
	h_graph_level = (bool*) malloc(sizeof(bool) * nb_nodes);
	h_graph_visited = (bool*) malloc(sizeof(bool) * nb_nodes);

		//initialize the memory of nodes
	h_graph_nodes[0].starting = 0;
	h_graph_nodes[0].no_of_edges = degrees[0];
	h_graph_level[0] = false;
	h_graph_visited[0] = false;
	for (unsigned int i = 1; i < nb_nodes; i++) {
		h_graph_nodes[i].starting = starting[i];
		h_graph_nodes[i].no_of_edges = degrees[i] - degrees[i-1];
		h_graph_level[i] = false;
		h_graph_visited[i] = false;
	}
	h_graph_level[start_position] = true;
	h_graph_visited[start_position] = true;

	//allocate memory for the result on host
	h_cost = (int*)malloc(sizeof(int) * nb_nodes);
	for (int i = 0; i < nb_nodes; i++) {
		h_cost[i] = -1;
	}
	h_cost[start_position] = 0;

	pthread_t pth[nb_nodes];
			
	do {
		d_over = false;
		for(int num = 0; num <nb_nodes; num = num+10) 
		{ 	
			for(int i = num; i < num_of_threads+num; i++){
				//cout << "i is " << i << endl;
				pthread_create(&pth[i],NULL, bfs_parallel,(void *)i);
			}

			for(int i=num;i<num_of_threads+num;i++) {
					pthread_join(pth[i], NULL);
			}			
		}
		cout << "Iteration" << endl;
	} while(d_over);
		
	//Store the result into a file
	cout<<"Write result file"<<endl;
	FILE* fpo = fopen("result.txt", "w");
	for (int i = 0; i < nb_nodes; i++) {
		fprintf(fpo, "(%d) cost:%d\n", i, h_cost[i]);
	}
	fclose(fpo);
}

void* bfs_parallel(void *n) {
	long i = (int) (long) n;
	//cout << "  kernel i is "<< i <<endl;

	if (i < nb_nodes && h_graph_level[i]) {
		//cout<<"Process thread"<< endl;
		h_graph_level[i] = false;		        
		for (int j = h_graph_nodes[i].starting; j < (h_graph_nodes[i].no_of_edges + h_graph_nodes[i].starting); j++) {
			int id = links[j];
			if (!h_graph_visited[id]) {
				h_graph_visited[id] = true;
						
				//calculate in which level the vertex is visited
				h_cost[id] = h_cost[i] + 1;
				h_graph_level[id] = true;
						
				//to make the loop continues
				d_over = true;
			}
		}
	}
	if (i == 3)
	{
		cout << "15152: " << h_graph_level[i] << " " << h_cost[i] << endl;
	}

	return NULL;
}
