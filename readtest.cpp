#include<iostream>
#include<fstream>
#include<stdio.h>
#include<stdlib.h>
#include<string>

using namespace std;

void bfsGraph(char *filename)
{
    int nb_nodes;
    ifstream finput;
    finput.open(filename, ios::in | ios::binary);

    finput.read((char *)&nb_nodes, 4);
    cout << nb_nodes << endl;

    int* degrees = new int[nb_nodes];
    for (int i = 0; i < nb_nodes; i++)
        degrees[i] = 0;

    finput.read((char*)degrees, nb_nodes * 4);
    cout << degrees[0];
    for (int i = 1; i < nb_nodes; i++)
        cout << " " << degrees[i];
    cout << endl;

    int nb_links = degrees[nb_nodes - 1];
    int* links = new int[nb_links];
    finput.read((char*)links, nb_links * 4);
    finput.close();
    cout << links[0];
    for (int i = 1; i < nb_links; i++)
        cout << " " << links[i];
    cout << endl;
}

int main(int argc, char *argv[])
{
    char *filename = argv[1];
    bfsGraph(filename);
    return 0;
}