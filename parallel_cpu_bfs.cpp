#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string.h>
#include <sstream>

#define N_NODES 16384

using namespace std;

vector<string> &split(const string &s, char delim, vector<string> &elems)
{
	stringstream ss(s);
	string item;
	while(getline(ss, item, delim))
	{
		elems.push_back(item);
	}
	return elems;
}

vector<string> split(const string &s, char delim)
{
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}

int main(int argc, char *argv[])
{
	if (argc != 3)
    {
		printf("Usage: %s <file> <num threads>\n",
               argv[0]);
		exit(EXIT_FAILURE);
	}

	int thread_count = atoi(argv[2]);
    char *filename = argv[1];
    
    ifstream file(filename);

	if (!file)
    {
		printf("Can't open file %s\n", argv[1]);
		exit(EXIT_FAILURE);
	}
    
    if (thread_count < 1)
    {
        printf("The thread count must be at least 1\n");
        exit(EXIT_FAILURE);
    }

    list<int> graph[N_NODES];
    string line;

    // skip header
    getline(file,line);
    int nodeNum = 0;
    int neighbor;

    // read file
    while(getline(file,line))
    {
		//list<int> adj;
		vector<string> elems = split(line, ',');
		vector<string> adjStrs = split(elems[1], ' ');

		// create adj list for this node
		// ASSUMES DIRECTIONAL GRAPH or file with reciprocated adj lists
		for (vector<string>::iterator it = adjStrs.begin(); it != adjStrs.end(); ++it)
		{
			string s = *it;
			neighbor = atoi(s.c_str());
			graph[nodeNum].push_back(neighbor);
		}
    }
    
    

	return 0;
}