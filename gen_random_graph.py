#! /usr/bin/python

from random import sample
from random import randint
from struct import pack
from sys import stdout

edges = 10000	#max edges per node (degree)
nodes = 100000	#total num nodes


# write out the number of nodes
b = pack("@i", nodes)
stdout.write(str(b))

# write out the cumulative degree of each node
first = randint(1,edges)
a = [first]
sum = first
b = pack("@i",sum)
stdout.write(str(b))

for i in range(1,nodes):						# range is [a,b)
	num_edges = randint(1,edges)
	sum += num_edges
	a.append(num_edges)
	b = pack("@i",sum)
	stdout.write(str(b))

for i in a:
	adj = sample(range(0,nodes),a[i])			# sample chooses without replacement
	for j in adj:
		# don't add itself to adj list
		if j != i:
			b = pack("@i",j)
			stdout.write(str(b))
		else:
			# try one more time to generate a unique neighbor
			s = randint(0,nodes)
			if not (s in adj):
				b = pack("@i",s)
				stdout.write(str(b))
print ""
