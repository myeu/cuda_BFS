#! /usr/bin/python

from random import sample
from random import randint

edges = 1000
nodes = 16384

print("node_id,adj_list")

line = ""
for i in range(0,nodes):						# range is [a,b)
	num_edges = randint(1,edges)
	line = str(i) + ","
	adj = sample(range(0,nodes),num_edges)		# sample chooses without replacement				
	for j in adj:
		# don't add itself to adj list
		if j != i:
			line += str(j) + " "
		else:
			# try one more time to generate a unique neighbor
			s = randint(0,nodes)
			if not (s in adj):
				line += str(s) + " "

	print(line)
