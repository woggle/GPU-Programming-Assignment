# Sample serial codes
This repository includes sample serial codes for the listed tasks for Homework 3.

It also has the Makefile setup to be able build the supplied serial codes.
For all of these serial codes, you can copy the supplied .cu file and make
corresponding your changes to your Makefile to use one of the supplied serial codes in your project.

# Breadth-First Search

[Wikipedia article](https://en.wikipedia.org/wiki/Breadth-first_search).

Files:
* `breadth_first_search.cu`: serial BFS implementation and verifier
* `bfs-tools/trivial-graph.txt`: example input file for breadth\_first\_search
* `bfs-tools/simplify_graph.py`: given a file with list of space seperated
  node#-node# pairs (representing edges), reformat the files in a format
  better suited for passing to the BFS tool. You can use this with the graphs
  for the [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/index.html)
  to use them to test your BFS.

Also, I have preformated large graphs in /bigtemp/cr4bd/graphs (on any cluster login machine).
I recomend testing performance on com-dblp.txt from that directory

The `breadth_first_search` binary which can be invoked like:
    
    ./breadth_first_search GRAPH -s START -k KERNEL_CODE

`GRAPH` is the name of a graph input file the `bfs-tools/trivial-graph.txt` we supply.
`START` is the number of the node to start the BFS at, by default 0.
`KERNEL_CODE` is a parameter passed to the GPU routine intended to allow you to easily
experiment with multiple versions of your GPU code like in Part 1 of the homework.

There are comments at the top of `breadth_first_search.cu` which should explain
the format of input passed to the `GPU_bfs` routine and the output expected from it.

The `GPU_bfs` routine is where you should primarily implement your code, but if you need
to change the supplied skeleton, that's up to you. You can look at the implementation of
`CPU_bfs` for reference.

# Forward substitution

[Wikipedia article](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution)

As a slight generalization this solves multiple systems of equations at once, solving
`AX = B` for a matrix X and B rather than solving `Ax = b` for a vector x and vector b.

Files:
* `forward_subst.cu`: serial forward substitue implementation and checking code

The `forward_subst` binary can be invoked like:
    
    ./forward_subst N M -k KERNEL_CODE

`N` is the number of rows and columns in A, `M` in the number of columns in `X` and `B`.
`KERNEL_CODE` is a parameter passed to the GPU routine intended to allow you to easily experiment
with multiple versions of your GPU code like in Part 1 of the homework.

Comments near the beginning of `forward_subst.cu` describe the input format to the `GPU_forward_subst`
routine enough to explain the layout of the matrices. You can look at the implementation
of `CPU_forward_subst` for reference.

Note that some care is taken to hopefully construct numerically stable examples, like would occur when this
routine was used with an LU decomposition. Without this, it's easy to suffer [catastraophic
cancellation](https://en.wikipedia.org/wiki/Loss_of_significance).

For debugging, rather than generate a random example, you can uncomment the `#define USE_EXAMPLE`
and edit the example around line 72. You can also uncomment the `#define PRINT_ANSWERS` to print
out what the CPU and GPU code compute.

# 2D convolution

[Wikipedia article](https://en.wikipedia.org/wiki/Kernel_%40image_processing%41).

We use the 'extend' choice from the wikipedia article to handle edges.

Files:
* `2d_convolve`: serial 2D convolution implementation and checking code

The `2d_convolve` binary can be invoked like:

    ./2d_convolve N K -k KERNEL_CODE

`N` is the number of rows and number of columns in the `image` matrix, `K` is the
number of rows and number of columns in the filter matrix. `K` must be odd.

Comments near the beginning of `2d_convolve.cu` describe the convolution task and the format
of A and the filter F in memory. You can also just look at the implementation of 
`CPU_convolve` for reference.
