#include <cassert>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include <cuda.h>
#include <sys/time.h>

/* 
Graph --- Representation of a directed graph.

Nodes are numbered starting with 0, to num_nodes - 1.

The edges for node i are stored starting at
    edge_destinations[edge_offsets[i]]
and continuing until and *excluding*
    edge_destinations[edge_offsets[i+1]]

Edges for a node will be in sorted order.


For example, the graph with the following edges:
    
    0 -> 1
    0 -> 2
    1 -> 2
    3 -> 4

would be represented as

    num_nodes = 5
    edge_offsets = {
        0, (index 0)
        2, (index 1)
        3, (index 2)
        3, (index 3)
        4, (index 4)
        4, (index 5)
    }
    edge_destinations = {
        1, (represents 0 to 1)
        2, (represents 0 to 2)
        2, (represents 1 to 2)
        4, (represents 3 to 4)
    }
*/
struct Graph {
    int num_nodes;
    int *edge_offsets;
    int *edge_destinations;
};

/*
CPU_bfs() --- reference implementation of a breadth-first search

arguments:
    Graph* theGraph --- the graph to search
    int starting_node --- the index of the node to start the search at
    int *output_bfs_tree --- output, giving the results of the search, described below

This performs a breadth-first search and outputs the result to output_bfs_tree.

If node i was not reached during the BFS, then output_bfs_tree[i] will be -1.

If node i was the starting_node, then output_bfs_tree[i] will be i.

If node i was reached during the BFS, then output_bfs_tree[i] will be the index of
the node from which it was first reached. If there are multiple possibilities,
then *any* of them is permissible.

For example, given the graph

    0 -> 1
    0 -> 2
    1 -> 2
    1 -> 6
    2 -> 3
    2 -> 6
    3 -> 0
    4 -> 5

then the after CPU_bfs(theGraph, 0, output_bfs_tree), the values in output_bfs_tree will be:
    
    output_bfs_tree[0] = 0
    output_bfs_tree[1] = 0
    output_bfs_tree[2] = 0
    output_bfs_tree[3] = 2
    output_bfs_tree[4] = -1
    output_bfs_tree[5] = -1
    output_bfs_tree[6] = 1  OR  output_bfs_tree[6] = 2 (either is permitted)

*/
void CPU_bfs(Graph* graph, int starting_node, int *output_bfs_tree);

void GPU_bfs(Graph* graph, int starting_node, int *output_bfs_tree, int kernel_code, float *kernel_time, float *transfer_time);

/*
Verify the result of a BFS is correct

Returns true if it is, false otherwise. Outputs a message to stderr
about the first discovered disagreement.
*/
bool verify_bfs(Graph* graph, int starting_node, int *output_bfs_tree);

/*
Load a graph from a file in starting_node<whitespace>ending_node pairs.

Nodes must be in numerical order and the edges for a node must be in
numerical order.

The file may also contain comments lines starting with '#' which
will be ignored.
*/
 
void load_graph(FILE *in, Graph *outGraph);

/* Timing utility functions */
float usToSec(long long time);
long long start_timer();
long long stop_timer(long long start_time, const char *name);
void die(const char *message);

// Main program
int main(int argc, char **argv) {

    //default kernel
    int kernel_code = 1;
    int starting_node = -1;
    
    // Parse vector length and kernel options
    const char *graph_file;
    if (argc >= 2) {
        graph_file = argv[1];
        for (int i = 2; i < argc; ++i) {
            if (i + 1 < argc) {
                if (0 == strcmp(argv[i], "-k")) {
                    kernel_code = atoi(argv[i + 1]);
                    ++i;
                } else if (0 == strcmp(argv[i], "-s")) {
                    starting_node = atoi(argv[i + 1]);
                    ++i;
                } else {
                    die("USAGE: ../breadth_first_search input_file -s starting_node -k kernel_code");
                }
            } else {
                die("USAGE: ../breadth_first_search input_file -s starting_node -k kernel_code");
            }
        }
    } else {
        die("USAGE: ../breadth_first_search input_file -s starting_node -k kernel_code");
    }

    Graph graph;
    FILE *in = fopen(graph_file, "r");
    if (!in) {
        die("Could not open input file");
    }
    load_graph(in, &graph);

    printf("loaded graph with %d nodes\n", graph.num_nodes);

    if (starting_node == -1) {
        /* find first node with at least one out edge */
        ++starting_node;
        while (graph.edge_offsets[starting_node] == graph.edge_offsets[starting_node + 1])
            ++starting_node;
    }

    printf("selected to start at node %d\n", starting_node);

    int *cpu_output_bfs_tree = (int*) malloc(graph.num_nodes * sizeof(int));
    int *gpu_output_bfs_tree = (int*) malloc(graph.num_nodes * sizeof(int));
    long long start_cpu = start_timer();
    CPU_bfs(&graph, starting_node, cpu_output_bfs_tree);
    long long CPU_time = stop_timer(start_cpu, "CPU version");

    if (!verify_bfs(&graph, starting_node, cpu_output_bfs_tree)) {
        fprintf(stderr, "CPU BFS produces INCORRECT RESULT!\n");
    }

#ifdef DEBUG
    fprintf(stderr, "first few parents are %d/%d/%d/%d/%d/%d\n",
        cpu_output_bfs_tree[0], cpu_output_bfs_tree[1],
        cpu_output_bfs_tree[2], cpu_output_bfs_tree[3],
        cpu_output_bfs_tree[4], cpu_output_bfs_tree[5]);
#endif

    float GPU_kernel_time = INFINITY;
    float transfer_time = INFINITY;
    long long start_gpu = start_timer();
    GPU_bfs(&graph, starting_node, gpu_output_bfs_tree, kernel_code, &GPU_kernel_time, &transfer_time);
    long long GPU_time = stop_timer(start_gpu, "GPU version");

    // Compute the speedup or slowdown
    //// Not including data transfer
    if (GPU_kernel_time > usToSec(CPU_time)) printf("\nCPU outperformed GPU kernel by %.2fx\n", (float) (GPU_kernel_time) / usToSec(CPU_time));
    else                     printf("\nGPU kernel outperformed CPU by %.2fx\n", (float) usToSec(CPU_time) / (float) GPU_kernel_time);

    //// Including data transfer
    if (GPU_time > CPU_time) printf("\nCPU outperformed GPU total runtime (including data transfer) by %.2fx\n", (float) GPU_time / (float) CPU_time);
    else                     printf("\nGPU total runtime (including data transfer) outperformed CPU by %.2fx\n", (float) CPU_time / (float) GPU_time);
    
    if (!verify_bfs(&graph, starting_node, gpu_output_bfs_tree)) {
        fprintf(stderr, "GPU BFS produces INCORRECT RESULT!\n");
    }

    cudaFree(cpu_output_bfs_tree);
    cudaFree(gpu_output_bfs_tree);
    cudaFree(graph.edge_offsets);
    cudaFree(graph.edge_destinations);
}

void GPU_bfs(Graph* graph, int starting_node, int *output_bfs_tree, int kernel_code, float *kernel_runtime, float *transfer_runtime) {
    for (int i = 0; i < graph->num_nodes; ++i) {
        output_bfs_tree[i] = -1;
    }
    output_bfs_tree[starting_node] = starting_node;
    // IMPLEMENT YOUR BFS HERE
}

void CPU_bfs(Graph* graph, int starting_node, int *output_bfs_tree) {
    char *visited;
    cudaMallocHost((void**) &visited, graph->num_nodes);
    for (int i = 0; i < graph->num_nodes; ++i) {
        output_bfs_tree[i] = -1;
        visited[i] = 0;
    }
    output_bfs_tree[starting_node] = starting_node;

    int *frontier;
    cudaMallocHost((void**) &frontier, sizeof(int) * graph->num_nodes);
    int frontier_start = 0;
    int frontier_end = 1;
    int next_frontier_end = 1;
    frontier[0] = starting_node;
    visited[starting_node] = 1;

    while (frontier_end > frontier_start) {
        for (int parent_index = frontier_start;
             parent_index < frontier_end;
             ++parent_index) {
            int parent = frontier[parent_index];
            for (int edge_index = graph->edge_offsets[parent];
                 edge_index < graph->edge_offsets[parent + 1];
                 ++edge_index) {
                int child = graph->edge_destinations[edge_index];
#ifdef DEBUG
                fprintf(stderr, "BFS: processing child %d of %d\n", 
                    child, parent);
#endif
                if (visited[child]) continue;
                visited[child] = 1;
                output_bfs_tree[child] = parent;
                frontier[next_frontier_end++] = child;
            }
        }
        frontier_start = frontier_end;
        frontier_end = next_frontier_end;
    }

    cudaFree(frontier);
    cudaFree(visited);
}

static bool next_pair(FILE *in, int *first, int *second) {
    for (;;) {
        char line[4096];
        char *result = fgets(line, sizeof line, in);
        if (!result) {
            return false;
        }
        if (line[strlen(line)] == '\n') {
            fprintf(stderr, "load_graph: excessively long line starting with [%s]\n", line);
            exit(EXIT_FAILURE);
        }

        if (line[0] == '#') {
            continue; // comment
        }
        if (sscanf(line, "%d %d", first, second) == 2) {
            return true;
        } else {
            fprintf(stderr, "load_graph: malformed line: [%s]\n", line);
            exit(EXIT_FAILURE);
        }
    }
}

void load_graph(FILE *in, Graph* graph) {
    /* First read file to determine sizez */
    int first, second;
    int max_node = 0;
    int num_edges = 0;
    while (next_pair(in, &first, &second)) {
        if (first > max_node) {
            max_node = first;
        }
        if (second > max_node) {
            max_node = second;
        }
        ++num_edges;
    }
    graph->num_nodes = max_node + 1;
    cudaMallocHost((void**) &graph->edge_offsets, sizeof(int) * (max_node + 1));
    cudaMallocHost((void**) &graph->edge_destinations, sizeof(int) * (num_edges));
    rewind(in);
    int last_first = 0, last_second = -1;
    graph->edge_offsets[0] = 0;
    graph->edge_offsets[1] = 0;
    while (next_pair(in, &first, &second)) {
        if (last_first < first) {
            last_second = -1;
        }
        while (last_first < first) {
            graph->edge_offsets[last_first + 2] = graph->edge_offsets[last_first + 1];
            ++last_first;
        }
        assert(second > last_second);
        graph->edge_destinations[
            graph->edge_offsets[first + 1]++
        ]  = second;
    }
}

bool verify_bfs(Graph* graph, int starting_node, int *output_bfs_tree) {
    int *distances;
    cudaMallocHost((void**) &distances, sizeof(int) * graph->num_nodes);
    for (int i = 0; i < graph->num_nodes; ++i) {
        distances[i] = -1;
    }
    distances[starting_node] = 0;
    /* first find the distances of all nodes to the starting node based
       on the BFS tree */
    for (int node = 0; node < graph->num_nodes; ++node) {
        /* special case for starting node */
        if (node == starting_node) {
            if (output_bfs_tree[node] != starting_node) {
                fprintf(stderr, "starting node %d linked to %d instead of self\n", 
                    starting_node, output_bfs_tree[node]);
                return false;
            }
            continue;
        }

        /* if the node is contained in the BFS tree, go up to the root and find the distance */
        if (output_bfs_tree[node] != -1) {
            int max_iterations = graph->num_nodes + 1;
            int saw_distance = 1;
            int parent = output_bfs_tree[node];
            if (parent >= graph->num_nodes) {
                fprintf(stderr, "node %d linked to impossible node %d\n", node, parent);
                goto out_failed;
            }
            if (distances[parent] != -1) {
                distances[node] = distances[parent] + 1;
            } else {
                while (saw_distance < max_iterations && parent != starting_node) {
                    parent = output_bfs_tree[parent];
                    if (parent == -1 || parent >= graph->num_nodes) {
                        fprintf(stderr, "node %d chains to impossible node %d in BFS tree (via %d)\n", node, parent, output_bfs_tree[node]);
                        return false;
                    }
                    ++saw_distance;
                }
                if (saw_distance == max_iterations) {
                    fprintf(stderr, "node %d is part of a cycle in the BFS tree\n", node);
                    return false;
                }
                distances[node] = saw_distance;
                distances[output_bfs_tree[node]] = saw_distance - 1;
            }
        }

    }

    /* now that distances are computed, for each node in the BFS tree check that
       (1) its parent actually has an edge to it
       (2) the distances of all its children are >= 1 + its distance
      */
    for (int node = 0; node < graph->num_nodes; ++node) {
        int parent = output_bfs_tree[node];
        if (parent == -1) {
            continue;
        }

        /* check for edge from parent to node */ 
        if (node != starting_node) {
            int low_edge_index = graph->edge_offsets[parent];
            int high_edge_index = graph->edge_offsets[parent + 1];
            int found_index = -1;
            /* binary search
               current valid range is [low_edge_index, high_edge_index)
             */
            while (low_edge_index < high_edge_index) {
                int midpoint = low_edge_index + (high_edge_index - low_edge_index) / 2;
                int midpoint_destination = graph->edge_destinations[midpoint];
                if (midpoint_destination == node) {
                    found_index = midpoint;
                    break;
                } else if (midpoint_destination > node) {
                    high_edge_index = midpoint;
                } else {
                    low_edge_index = midpoint + 1;
                }
            }
            if (found_index == -1) {
                fprintf(stderr, "node %d has parent %d in BFS tree, but no %d->%d edge in graph\n",
                    node, parent, parent, node);
                goto out_failed;
            }
            
            /* if this isn't true, the code in the previous loop is probably broken */
            if (distances[node] != distances[parent] + 1) {
                fprintf(stderr, "inconsistent distance for node %d and parent %d\n", node, parent);
                goto out_failed;
            }
        }

        /* check distances of children */
        for (int edge_index = graph->edge_offsets[node];
             edge_index < graph->edge_offsets[node + 1];
             ++edge_index) {
            int child = graph->edge_destinations[edge_index];
            if (distances[child] > distances[node] + 1) {
                fprintf(stderr, "child %d of %d is at distance %d via %d, but could be at distance %d via %d\n",
                    child, node, distances[child],
                    output_bfs_tree[child],
                    distances[node] + 1,
                    node);
                goto out_failed;
            } else if (distances[child] == -1) {
                fprintf(stderr, "child %d of %d is not in BFS tree, but %d is\n",
                    child, node, node);
                goto out_failed;
            }
        }
    }

    cudaFree(distances);
    return true;
    
out_failed:
    cudaFree(distances);
    return false;
}

// Returns the current time in microseconds
long long start_timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// converts a long long ns value to float seconds
float usToSec(long long time) {
    return ((float)time)/(1000000);
}

// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, const char *name) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
    float elapsed = usToSec(end_time - start_time);
    printf("%s: %.5f sec\n", name, elapsed);
    return end_time - start_time;
}

// Prints the specified message and quits
void die(const char *message) {
    printf("%s\n", message);
    exit(1);
}
