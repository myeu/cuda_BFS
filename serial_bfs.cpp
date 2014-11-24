#include <iostream>
#include <list>

using namespace std;

#define NUM_EDGES_PER_NODE 600


class Graph 
{
private:
	int n;		//num of V
	int **A;	// stores edges between two verts
	int *C;		// stores num edges for a node
public:
	Graph(int size=2);
	~Graph();
	int* getEdges(int u);
	int getNumEdges(int u);
	void addEdge(int u, int ui, int v, int vi);
	void BFS(int s);
};

Graph::Graph(int size) 
{
	if (size < 2) n = 2;
	else n = size;

	A = new int*[n];

	for (int i = 0; i < n; ++i)
		A[i] = new int[NUM_EDGES_PER_NODE];

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < NUM_EDGES_PER_NODE; j++)
			A[i][j] = 0;
	}

	C = new int[n];

	for (int i = 0; i < n; i++)
		C[i] = 0;
}

Graph::~Graph()
{
	for (int i = 0; i < n; ++i)
		delete [] A[i];
	delete [] A;
	delete [] C;
}

int* Graph::getEdges(int u)
{
	return A[u];
}

int Graph::getNumEdges(int u)
{
	return C[u];
}

void Graph::addEdge(int u, int ui, int v, int vi)
{
	A[u][ui] = v;
	A[v][vi] = u;
	C[u] += 1;
	C[v] += 1;
}

void Graph::BFS(int s) 
{
	bool *explored = new bool[n+1];

	for (int i = 1; i <= n; ++i)
		explored[i] = false;

	list<int> queue;
	queue.push_back(s);
	explored[s] = true;

	int s_edges, u; 

	while (!queue.empty())
	{
		s = queue.front();
		queue.pop_front();

		cout << "explored " << s << endl;

		s_edges = getNumEdges(s);

		for (int w = 0; w < s_edges; w++)
		{
			u = A[s][w];
			if(!explored[u])
			{
				queue.push_back(u);
				explored[u] = true;
			}
		}
	}
}

void mini_example(Graph &g)
{
	g.addEdge(1, 0, 2, 0); 
    g.addEdge(1, 1, 3, 0);
    g.addEdge(2, 1, 4, 0); 
    g.addEdge(3, 1, 4, 1);
    g.addEdge(3, 2, 6, 0); 
    g.addEdge(4, 2, 7, 0);
    g.addEdge(5, 0, 6, 1); 
    g.addEdge(5, 1, 7, 1);

    g.addEdge(9, 0, 0, 0); 
    g.addEdge(5, 2, 8, 0);
    g.addEdge(9, 1, 7, 2); 
    g.addEdge(4, 3, 10, 0);
    g.addEdge(11, 0, 8, 1); 
    g.addEdge(11, 1, 0, 1);
    g.addEdge(10, 1, 5, 3); 
    g.addEdge(5, 4, 2, 2);
}

int main()
{
	Graph g(12);
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

    g.BFS(5);
}

