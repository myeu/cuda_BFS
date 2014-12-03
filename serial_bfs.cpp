#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <list>
#include <fstream>
#include <string>
#include <sys/time.h>

using namespace std;

#define NUM_EDGES_PER_NODE 1000


class Graph 
{
private:
	int n;		//num of V
	int **A;	// stores edges between two verts
	int *C;		// stores num edges for a node
public:
	Graph(int size=2);
	//~Graph();
	int* getEdges(int u);
	int getNumEdges(int u);
	void addEdge(int u, int ui, int v);
	int BFS(int s, char *outFile);
};

Graph::Graph(int size) 
{
	if (size < 2) 
		n = 2;
	else 
		n = size;

	A = new int*[n];	

	for (int i = 0; i < n; i++)
	{
		A[i] = new int[NUM_EDGES_PER_NODE];
		for (int j = 0; j < NUM_EDGES_PER_NODE; j++)
			A[i][j] = 0;
	}

	C = new int[n];

	for (int i = 0; i < n; i++)
		C[i] = 0;
}

/*Graph::~Graph()
{
	cerr << n << endl;
	for (int i = 0; i < n; ++i)
		delete [] A[i];
	delete [] A;
	delete [] C;
}*/

int* Graph::getEdges(int u)
{
	return A[u];
}

int Graph::getNumEdges(int u)
{
	return C[u];
}

void Graph::addEdge(int u, int ui, int v)
{
	A[u][ui] = v;
	C[u] += 1;
}

int Graph::BFS(int s, char* outFile) 
{
	bool *explored = new bool[n];
	int *cost = new int[n];

	for (int i = 1; i < n; ++i)
	{
		explored[i] = false;
		cost[i] = -1;
	}

	list<int> queue;
	queue.push_back(s);
	explored[s] = true;
	cost[s] = 0;

	int s_edges, u, new_cost;
	int max_cost = 0; 

	while (!queue.empty())
	{
		s = queue.front();
		queue.pop_front();

		//cout << "explored " << s << endl;

		s_edges = getNumEdges(s);

		for (int w = 0; w < s_edges; w++)
		{
			u = A[s][w];
			if(!explored[u])
			{
				queue.push_back(u);
				explored[u] = true;
				new_cost = cost[s] + 1;
				cost[u] = new_cost;
				if (new_cost > max_cost)
					max_cost = new_cost;
			}
		}
	}

	/*for (int i = 0; i < n; i++)
	{
		printf("(%d) cost:%d\n", i, cost[i]);
	}*/

	FILE *fpo = fopen(outFile, "w");
	for (int i = 0; i < n; i++)
	{
		fprintf(fpo, "(%d) cost:%d\n", i, cost[i]);
	}
	fclose(fpo);

	delete [] explored;
	delete [] cost;

	return max_cost;
}

void mini_example(Graph &g)
{
	g.addEdge(0, 0, 1);
	g.addEdge(1, 0, 2); 
    g.addEdge(1, 1, 3);
    g.addEdge(2, 0, 4); 
    g.addEdge(3, 0, 4);
    g.addEdge(3, 1, 6); 
    g.addEdge(4, 0, 7);
    g.addEdge(4, 1, 10);
    g.addEdge(5, 0, 6); 
    g.addEdge(5, 1, 7);
    g.addEdge(5, 2, 8);
	g.addEdge(5, 3, 2);
    g.addEdge(6, 0, 2);
    g.addEdge(6, 1, 10);
    g.addEdge(7, 0, 11);
    g.addEdge(7, 1, 8);
    g.addEdge(7, 2, 3);
    g.addEdge(8, 0, 9);
    g.addEdge(8, 1, 3);
    g.addEdge(8, 2, 11);
    g.addEdge(9, 0, 0); 
    g.addEdge(9, 1, 7); 
    g.addEdge(10, 0, 5); 
    g.addEdge(11, 0, 8); 
    g.addEdge(11, 1, 0);    
}

void readFile(char *filename, Graph g)
{
	int nb_nodes;
    ifstream finput;
    finput.open(filename, ios::in | ios::binary);

    // Read the number of nodes
    finput.read((char *)&nb_nodes, 4);

    // Read cumulative degrees, 4 bytes per node
	int *degrees = new int[nb_nodes];
	finput.read((char*) degrees, nb_nodes * 4);

	// Create starting array based on degrees
	int *starting = new int[nb_nodes];
	//memset(starting, 0, sizeof(int) * nb_nodes);
	for (int i = 1; i < nb_nodes; i++)
	{
		starting[i] = degrees[i - 1];
	}

	// Read links, 4 bytes per link
	int nb_links = degrees[nb_nodes - 1];
	int *links = new int[nb_links];
	finput.read((char*) links, nb_links * 4);

	finput.close();
	// Create graph nodes by adding edges
	for (int i=0; i < nb_nodes; i++)
	{
		for (int j = starting[i]; j < degrees[i]; j++)
		{
			// edge from node i, the edge num for this node, the neighbor node
			//cout << i << ": " << j << ", " << j - starting[i] << ", " << links[j] << " [" << degrees[i] << "]" << endl;
			g.addEdge(i, j - starting[i], links[j]);
		}
	}

	delete [] degrees;
	delete [] starting;
	delete [] links;
}



int main(int argc, char **argv)
{
	char *filename = argv[1];
	char *outFile = argv[2];
	int numNodes = atoi(argv[3]);
	int startNode = rand() % numNodes;

    Graph g(numNodes);
    readFile(filename, g);

    /*Graph g(12);
    mini_example(g);

    for (int j = 0; j < 12; j ++)
    {
    	int* tmp = g.getEdges(j);
    	int tmpN = g.getNumEdges(j);

    	cout << "Edges of vertex ";
    	cout << j << ": ";
    	
    	for (int i = 0; i < tmpN; i++)
    	{
    		cout << tmp[i] << ", ";
    	}
    	cout << endl;
    }
    cout << endl;
	*/

    int max_cost = 0;
    struct timeval start, end;    
	gettimeofday(&start, NULL);

    max_cost = g.BFS(startNode, outFile);

    gettimeofday(&end, NULL);
	printf("%ld %d\n",
           (end.tv_sec * 1000000 + end.tv_usec)
           - (start.tv_sec * 1000000 + start.tv_usec), max_cost);
}

