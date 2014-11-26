#include <stlib.h>
#include <limits>
#include <list>

class Node
{
private:
	long dist;
	list<int> adj;
public:
	Node(long d = LONG_MAX);
	~Node();
	long getDist();
	void addAdj(int a);
};