#! /usr/bin/python

from struct import pack
from sys import stdout
from sys import stderr
import argparse

parser = argparse.ArgumentParser(description='Read a .graph file and print binary format to stdout')
parser.add_argument('graphFile', metavar='f', help='file of .graph format')

args = parser.parse_args()

graphFile = open(args.graphFile, "r")

line = graphFile.readline()
stderr.write(line)
toks = line.split()
nb_nodes = int(toks[0])
nb_links = int(toks[1])

degrees = []
edges = []
d = 0

for line in graphFile:
	toks = line.split()
	for e in toks:
		d += 1
		edges.append(int(e))
	degrees.append(d)

graphFile.close()

b = pack("@i", nb_nodes)
stdout.write(str(b))

for d in degrees:
	b = pack("@i", d)
	stdout.write(str(b))

for e in edges:
	b = pack("@i", e)
	stdout.write(str(e))

fnode_edges = ""
for i in range(degrees[0],degrees[1]):
	fnode_edges += (" " + str(edges[i]))
stderr.write("1 " + str(degrees[1]) + fnode_edges + "\n")

stderr.write("nb_links: " + str(nb_links) + " len(edges): " + str(len(edges)) + "\n")



