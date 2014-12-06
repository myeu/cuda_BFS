/*
 * main.cpp
 *
 *  Created on: Nov 27, 2014
 *      Author: rtamaskar
 */
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include <pthread.h>
#include <cstring>
#include <sys/time.h>
#include <unistd.h>

using namespace std;

# define num_of_threads 10

void bfsGraph(char* filename, char *outFile);

void* bfs_parallel(void *n);

typedef struct Node {
	int starting;
	int no_of_edges;
}Node;

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		printf("Usage: %s <input file> <output file>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	char *filename = argv[1];
	char *outFile = argv[2];

	bfsGraph(filename, outFile);
}

int nb_nodes;
Node* h_graph_nodes;
int* degrees;
int* starting;
int* links;
int* h_cost;
bool d_over;
pthread_mutex_t the_mutex;

void bfsGraph(char* filename, char *outFile) {
	ifstream finput;
	pthread_mutex_init(&the_mutex, NULL);
	finput.open(filename, ios::in | ios::binary);

	//Read number of nodes on 4 bytes
	finput.read((char*)&nb_nodes, sizeof(int));

	int start_position = rand() % nb_nodes;

	if (start_position < 0 || start_position > nb_nodes) {
		return;
	}
	
	//Read cumulative degrees
	degrees = new int[nb_nodes];
	finput.read((char*)degrees, nb_nodes * sizeof(int));

	starting = new int[nb_nodes];
	memset(starting, 0, sizeof(int) * nb_nodes);
	for (int i = 1; i < nb_nodes; i++) {
		starting[i] = degrees[i - 1];
	}

	//Read links
	int nb_links = degrees[nb_nodes - 1];
	links = new int[nb_links];
	finput.read((char*)links, nb_links * sizeof(int));

	finput.close();

	cout << "read file, nb_nodes: " << nb_nodes << " nb_links: " << nb_links << endl;

	//allocate host memory
	h_graph_nodes = (Node*) malloc(sizeof(Node) * nb_nodes);

	//initialize the memory of nodes
	h_graph_nodes[0].starting = 0;
	h_graph_nodes[0].no_of_edges = degrees[0];
	for (unsigned int i = 1; i < nb_nodes; i++) {
		h_graph_nodes[i].starting = starting[i];
		h_graph_nodes[i].no_of_edges = degrees[i] - degrees[i-1];
	}

	//allocate memory for the result on host
	h_cost = (int*)malloc(sizeof(int) * nb_nodes);

	for (int i = 0; i < nb_nodes; i++) {
		h_cost[i] = -1;
	}

	h_cost[start_position] = 0;
	cout << "starting pos: " << start_position << endl;

	int iteration = 0;
	int *threadInfo;
	
	//pthread_t pth[num_of_threads];
	
	struct timeval start, end;   
	gettimeofday(&start, NULL);
	unsigned int microseconds = 1000;

	do {
		d_over = false;
		for(int num = 0; num < nb_nodes; num = num + num_of_threads) 
		{ 	
			pthread_t pth[num_of_threads];
			for(int i = num; i < num_of_threads+num; i++){
				threadInfo = (int *) malloc(sizeof(int) * 2);
				threadInfo[0] = iteration;
				threadInfo[1] = i;
				//cout << "global tid: " << i << " local id: " << (i - num) << endl;
				pthread_create(&pth[i - num],NULL, bfs_parallel,(void *) threadInfo);
				usleep(microseconds);
			}


			for(int i = num;i < num_of_threads; i++) {
				pthread_join(pth[i], NULL);
			}
		}
		iteration++;
	} while(d_over);

	gettimeofday(&end, NULL);
	printf("%ld %d\n",
           (end.tv_sec * 1000000 + end.tv_usec)
           - (start.tv_sec * 1000000 + start.tv_usec), iteration);
		
	//Store the result into a file

	//cout << "filename: " << filename << endl;

	FILE* fpo;
	if ((fpo = fopen(outFile, "w")) == NULL)
    {
		printf("Can't open file %s\n", outFile);
		exit(EXIT_FAILURE);
	}
	
	for (int i = 0; i < nb_nodes; i++) {
		//cout<< i << " " << h_cost[i] <<endl;
		fprintf(fpo, "(%d) cost:%d\n", i, h_cost[i]);
	}
	fclose(fpo);
}

void* bfs_parallel(void *info) {
	int *myInfo = (int *) info;
	int level = (int) myInfo[0];
	int i = (int) myInfo[1];
	int not_set = -1;

	//cerr << i << endl;

	if (i < nb_nodes && h_cost[i] == level) { //short circuts if i is out of bounds, cost[i] is safe
		//cout<<"Process thread"<< endl;
		for (int j = h_graph_nodes[i].starting; j < (h_graph_nodes[i].no_of_edges + h_graph_nodes[i].starting); j++) 
		{
			int id = links[j];
			cerr << "# " << h_cost[id] << endl;
			if (h_cost[id] == not_set) {
						
				//calculate in which level the vertex is visited
				h_cost[id] = h_cost[i] + 1;
						
				//to make the loop continues
				d_over = true;
			}
		}
	}
}
