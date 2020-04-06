# Growing Neural Cellular Automata - Task Experiments  

Based on the original work (Growing Neural Cellular Automata)[https://distill.pub/2020/growing-ca/] by Alexander Mordvintsev et al.  
  
The writing on this page is adapted from posts and discussion in the distill slack group.  

### Fun Pictures
A single network trained to converge to multiple target outputs specified by control channels [source](basic_large.py):  
  
![growth](https://i.imgur.com/vjrqwF2.gif)
  
### Visualizing hidden states:    
  
![hidden](https://i.imgur.com/2ApfNM3.gif)  

### Computational Tasks  
  
It is an open question what kind of computational tasks these systems can be optimized to perform. A simple building block for more complex problems, is copying a random input matrix from one location to another. From a high level this might seem like a trivial task. If you were building a physical circuit it would be as simple as connecting all the inputs directly to the outputs with wires. However, given the constraints that this is a homogeneous computational medium, communication is only allowed between immediate neighbors, and there is a 50% chance of failure every time you try to post a message, it's actually not so straightforward.
With some patience it is possible for the CA to learn this with 4x4 and 8x8 matrices! It was easier to make 4x4 work than 8x8, larger that 12x12 have not been tried for this experiment yet. It seems this system must need to execute some kind of bucket brigade algorithm in order to move the input data to the output. When visualizing the hidden states at high framerate, you can actually see the elements of the input matrix being consumed and streaming across the grid.
To clarify - the model is not trained to copy any particular matrices. Every run of the simulation has a new random matrix of floats 0-1, and the CA must copy it regardless of its content. The first image is the input/output channel, the next three span the other 9 hidden channels.  
  
### Copying a matrix from one location to another:  
[source](CA_tasks/CA_tasks_copy_1.py)  
  
![copy](https://i.imgur.com/oHirFid.gif)

### Matrix Multiplication
The next computational task tried is matrix-matrix multiplication. This requires moving elements from each input matrix, matching corresponding elements, multiplying them and accumulating results. So far up to 22x22 matrices have been trained. As input, uniform noise is replaced with contours of fractal (pink) noise. This distribution of frequencies creates input/output features which are more visually recognizable, and also makes the task easier to learn by rewarding incremental progress.
Here are some results. The lower left and upper right are the input matrices, while the lower right is the CAs output area, and the upper left is ground truth (overlayed at the end of the simulation)

<img src="https://i.imgur.com/dd9BVEq.png" width="400" height="800">

The way the CA implements this algorithm is similar to the systolic arrays in a TPU's matrix multiplication unit [Analysis of matrix multiplication in TPUs](https://medium.com/@CPLu/should-we-all-embrace-systolic-array-df3830f193dc). This seems natural, as both are grids of homogeneous units which communicate through their neighbors.
The next experiment is trying to capture the movement of information through the system. Here is one attempt, visualizing the 24 hidden channels (as 8 rgb images) side by side with the output channel. To track motion, I've found its better to visualize the change (residual) of the cells at each step. Due to the stochastic updating however these changes are very noisy and brief. To make patterns over time easier to see, the change is measured from an exponential moving average rather than just the state from the previous step. This creates "tracers" for each cell update. Some channels seem to exhibit mostly downward or rightward motion, and some are more of a mixture. The final result can clearly be seen filling in diagonally from the upper left corner of the output, similar as would be expected of a systolic array. Some of the most interesting patterns are grid like structures which fill the input matrices space near the near of the simulation. The reason for this is unknown at this time. The initial wave propagating up to the right from the center is an artifact of pretraining while allowing wrapping around the edges, which was later disabled. 

### Matrix Multiplication Visualization:  
[source](CA_tasks/CA_tasks_matmul.py)  
  
![matmul](https://i.imgur.com/3CD5IX7.gif)  
  
### Principal components of hidden channels:  
  
![matmul_pca](https://i.imgur.com/o9U0IWY.gif)  
  
