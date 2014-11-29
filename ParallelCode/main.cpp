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
	char*filename = "random_adj_graph_directed_binary.csv";
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
	    
//	    nb_nodes = 7;

	    //Read cumulative degrees
	    degrees = new int[nb_nodes];
	    finput.read((char*)degrees, nb_nodes * 4);
            /*int arr_degree[7] = {2,4,6,7,8,9,10};
            for(int k = 0; k <nb_nodes; k++) {
		degrees[k] = arr_degree[k];
	    }*/
	    starting = new int[nb_nodes];
	    memset(starting, 0, sizeof(int) * nb_nodes);
	    for (int i = 1; i < nb_nodes; i++) {
	        starting[i] = degrees[i - 1];
	    }

	    //Read links
	    int nb_links = degrees[nb_nodes - 1];
	    links = new int[nb_links];
	    finput.read((char*)links, nb_links * 4);
	   /* int arr_link[10] = {1,2,3,4,5,6,1,1,2,2};
            for(int k = 0; k <nb_links; k++) {
		links[k] = arr_link[k];
	    }*/

	    finput.close();

	    cout<<"Number of nodes are"<<nb_nodes<<endl;

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

	        //allocate memory for the result on host
			h_cost = (int*)malloc(sizeof(int) * nb_nodes);
			for (int i = 0; i < nb_nodes; i++) {
				h_cost[i] = -1;
			}
			h_cost[start_position] = 0;

	        pthread_t pth[num_of_threads];
		
	        do {
			
			d_over = false;
			for(int i=0;i<num_of_threads;i++){
					pthread_mutex_lock(&the_mutex);
	        			pthread_create(&pth[i],NULL, bfs_parallel,(void *)&i);
					sleep(3);
					pthread_mutex_unlock(&the_mutex);
			}
                }while(d_over);

	        for(int i=0;i<num_of_threads;i++)
	        	    pthread_join(pth[i], NULL);

		//Store the result into a file
    		FILE* fpo = fopen("result.txt", "w");
    		for (int i = 0; i < nb_nodes; i++) {
        		fprintf(fpo, "(%d) cost:%d\n", i, h_cost[i]);
    		}
            	fclose(fpo);

    //cleanup memory
//    free(h_graph_nodes);
//    free(links);
//    free(h_graph_level);
//    free(h_graph_visited);
//    free(h_cost);
}

void* bfs_parallel(void *n) {
	int thread_id = *(int*)n;
	int m = nb_nodes/num_of_threads;

	int first = thread_id*m;
        int last;
	if(num_of_threads > nb_nodes)
		last = nb_nodes;
        else
		last = first + m;

	for(int i = first; i<last; i++) {
		if (i < last && h_graph_level[i]) {
		        h_graph_level[i] = false;
		        h_graph_visited[i] = true;
		        for (int j = h_graph_nodes[i].starting; j < (h_graph_nodes[i].no_of_edges + h_graph_nodes[i].starting); j++) {
		            int id = links[j];
		            if (!h_graph_visited[id]) {
		                //calculate in which level the vertex is visited
		                h_cost[id] = h_cost[i] + 1;
		                h_graph_level[id] = true;
		                //to make the loop continues
		                d_over = true;
		            }
		        }
		    }
	}

	return NULL;
}
