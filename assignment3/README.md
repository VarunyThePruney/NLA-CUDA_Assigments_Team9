# Assignment 3 – Jacobi and Seidel Iterative Methods

SE24UCSE242 – Varun Sai Golakoti  
SE24UCSE244 – Konakalla Sai Shloka

[Assignment 3 – GitHub Repository](https://github.com/VarunyThePruney/NLA-CUDA_Assigments_Team9/tree/main/assignment3)

---

# Jacobi Iterative Method

The Jacobi method solves a system of linear equations \(Ax=b\) by repeatedly calculating approximations of the solution. Every value in a new iteration is calculated using values from the previous iteration. This makes the calculations independent and suitable for parallel execution on a GPU.

### Main Program Logic

The program follows three main steps:

1. **Matrix Sizes:** Reads the three input matrices of sizes 1024 × 1024, 2048 × 2048, and 4096 × 4096.
2. **CUDA Configurations:** Tests four configurations: 4 blocks × 32 threads, 8 × 64, 16 × 128, and 32 × 256. The initial solution is reset for each configuration.
3. **Iterations:** Executes the jacobi kernel, checks the difference between successive solutions, and repeats until the maximum difference is less than 10^-6.

The right-hand-side vector is initialized with all ones, while the initial solution vector is initialized with all zeros.

### Reading the Matrix

The readMatrix() function reads the matrix size and its elements from the input file similar to assignemnt 2's function. The matrix is stored in row-major order, where each element is accessed using:

A[i * n + j] &emsp;(A[i][j])

### Jacobi Calculation

For each unknown, the new value is calculated using:

x[i] = (b[i] - sum of A[i][j] \* x[j]) / A[i][i]

All calculations use values from the previous iteration. The new solution is stored separately and becomes the input for the next iteration.

### CUDA Implementation

The CUDA kernel uses a grid-stride loop to distribute the matrix rows among the available GPU threads. If there are more rows than threads, each thread processes multiple rows.

After each kernel execution, the new solution is copied to the CPU to check convergence. The old and new solution vectors are then swapped before the next iteration.

### Convergence and Timing

The program calculates the maximum difference between successive solution vectors:

max |x_new[i] - x[i]| < 0.000001

The iterations stop when maxDiff < 1e-6.

CUDA Events are used to measure the accumulated GPU kernel execution time, recorded as TGPU in seconds. The results are stored in results_gpu.csv.

---

# Seidel Iterative Method

The Seidel method also solves a system of linear equations \(Ax=b\). However, unlike jacobi, it immediately uses newly calculated values when calculating subsequent unknowns. This can reduce the number of iterations required for convergence, but introduces dependencies between calculations which makes parallelization more difficult

### Main Program Logic

The program reads the input matrices, initializes the right-hand-side vector with ones and the solution vector with zeros, and performs repeated Seidel iterations.

For the GPU implementation, four CUDA configurations are tested: 4 × 32, 8 × 64, 16 × 128, and 32 × 256.

### Seidel Calculation

The update equation is:

x[i] = (b[i] - sum of A[i][j] \* x[j]) / A[i][i]

The method uses newly updated values for earlier rows and previous iteration values for later rows.

The solution is updated in place, meaning that each new value is immediately available for subsequent calculations.

### CUDA Implementation

Unlike Jacobi, Seidel has dependencies between successive row calculations. Therefore, the implemented GPU kernel uses only one active CUDA thread to perform the calculations sequentially.

Although the program tests different block and thread configurations, the actual seidel calculation is performed by that single thread.

### Convergence and Timing

The program checks the maximum difference between the current and previous solution vectors:

max |x_new[i] - x[i]| < 0.000001

The iterations stop when maxDiff < 1e-6.

CUDA Events measure the accumulated GPU kernel execution time in seconds, recorded as TGPU. The results are stored in results_gpu_csv.

---

# Analysis

The Jacobi method takes thousands of iterations to converge, whereas seidel requires only 5–6 iterations for the given input matrices. This is because seidel immediately uses newly calculated values, allowing it to converge much faster on these matrices.

Jacobi is more suitable for GPU parallelization because its row calculations are independent. Increasing the number of blocks and threads generally improves its execution time, although the best configuration depends on the matrix size.

Seidel is more difficult to parallelize because each calculation depends on previous values. In this implementation, only one GPU thread performs the calculations, so changing the block and thread configurations has very little effect on execution time. Thats why CPU time is much faster for siedel, as cpu can perform the single row's math faster than the gpu.

When comparing the GPU execution times, Jacobi is faster for the 1024 × 1024 and 2048 × 2048 matrices in the tested configurations. For the 4096 × 4096 matrix, the execution times of both methods are relatively close.

Overall, seidel requires significantly fewer iterations, while jacobi makes better use of GPU parallelism. The actual execution time depends on both the number of iterations and the computational cost of each iteration.
