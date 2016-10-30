#!/usr/bin/env python2

# This program reformats a graph to be suitable for our breadth_first_search.cu
# This takes a list of source_vertex, destination_vertex pairs and:
# - renumbers them so nodes with no edges are omitted
# - sorts the edges
# - optionally turns undirected graphs in directed graphs with edges in both directions

# Usage:
#   python simplify_graph.py input_graph.txt >output_graph.txt
# or, for an undirected graph:
#   python simplify_graph.py --undirected input_graph.txt >output_graph.txt

from __future__ import print_function
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=file, default=sys.stdin)
parser.add_argument('--undirected', default=False, action='store_true')

args = parser.parse_args()

mapping = {}
edges = {}

all_seen = set()

for line in args.input_file.readlines():
    if line.startswith('#'):
        continue
    first, second = line.split()
    first = int(first)
    second = int(second)
    if first not in edges:
        edges[first] = []
    edges[first].append(second)
    if args.undirected:
        if second not in edges:
            edges[second] = []
        edges[second].append(first)
    all_seen.add(first)
    all_seen.add(second)

for i, node in enumerate(sorted(list(all_seen))):
    mapping[node] = i

for node in sorted(edges.keys()):
    for dest in sorted(edges[node]):
        print("{} {}".format(mapping[node], mapping[dest]))


