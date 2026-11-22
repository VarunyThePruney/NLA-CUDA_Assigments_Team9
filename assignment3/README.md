# Assignment 3 – Jacobi and Gauss–Seidel Iterative Methods

SE24UCSE242 - Varun Sai Golakoti  
SE24UCSE244 - Konakalla Sai Shloka

---

[Assignment 3 – GitHub Repository](https://github.com/VarunyThePruney/NLA-CUDA_Assigments_Team9/tree/main/assignment3)

# Jacobi Iterative Method

The Jacobi method solves a system of linear equations **Ax = b** by repeatedly approximating each component of the solution. Every updated value in an iteration is calculated using values from the **previous** iteration. This makes its row computations independent and suitable for CUDA parallelization.

### Main Program Logic:

The provided CUDA program has three main loops:

1. **Sizes Loop:** Iterates over input matrices of sizes **1024 × 1024**, **2048 × 2048**, and **4096 × 4096**, stored in `Cholesky_Input_1024x1024.txt`, `Cholesky_Input_2048x2048.txt`, and `Cholesky_Input_4096x4096.txt`. Despite their filenames, these are the input matrices used by the Jacobi program. The program reads each matrix, allocates CPU and GPU arrays, and initializes the right-hand-side vector `b` to all ones and the initial guess `x` to all zeros.
2. **CUDA Configuration Loop:** Tests four paired configurations: **4 blocks × 32 threads**, **8 × 64**, **16 × 128**, and **32 × 256**. The solution is reset to zero for each configuration.
3. **Iterations Loop:** Launches `jacobi()`, copies the new solution back to the CPU, calculates the maximum component-wise difference from the previous solution, swaps the old and new solution buffers, and repeats until the difference is less than `1e-6`.

### Reading Matrix:

`readMatrix()` opens the input file, reads the matrix order `n` from the first line, allocates an `n × n` `double` array, and reads its entries. The matrix uses **row-major storage**, so the entry at row `i`, column `j` is accessed as:

```c
A[i * n + j]
```

The vector `b` is not read from the input files; the program sets each `b[i] = 1`.

### Jacobi Logic:

For the linear system **Ax = b**, the update for the `i`-th unknown is:

\[
x_i^{(k+1)} = \frac{b_i - \sum_{j \ne i} A_{ij} x_j^{(k)}}{A_{ii}}
\]

Each update uses only the old solution vector, so no row depends on the newly calculated value of another row in the same iteration.

The kernel starts with `sum = b[i]`, subtracts `A[i * n + j] * x[j]` for all `j != i`, and stores the result divided by the diagonal element in `x_new[i]`.

### CUDA GPU Logic:

The GPU kernel computes a starting row and a grid-wide stride:

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (; i < n; i += stride) {
    // Compute x_new[i] using the previous vector x
}
```

This **grid-stride loop** allows one GPU thread to process several rows if there are more rows than launched threads. For example, 4 blocks of 32 threads launch 128 threads, which collectively cover all rows by stepping through the matrix at intervals of 128.

The CPU copies `A` and `b` to GPU memory once per matrix. At the start of each configuration, it copies the zero initial guess into `d_x`. After each kernel execution, it copies `d_x_new` back to the CPU to check convergence. It then swaps `d_x` with `d_x_new` so that the next kernel reads the latest GPU solution without copying the full vector back to the GPU.

### Convergence and Stopping Condition:

The program computes:

\[
\text{maxDiff} = \max_i \left|x_i^{(k+1)} - x_i^{(k)}\right|
\]

It stops when `maxDiff < 1e-6`. This is a **difference-between-iterates** test, not a direct check of the residual `||Ax - b||`. Jacobi convergence is not guaranteed for an arbitrary input matrix; properties such as strict diagonal dominance are common sufficient conditions. The code also assumes nonzero diagonal entries and does not impose a maximum iteration count.

### Timing and Results:

The program uses CUDA events to time each `jacobi()` kernel launch and accumulates the elapsed time in seconds as `TGPU`:

```c
cudaEventRecord(start);
jacobi<<<B, T>>>(d_A, d_b, d_x, d_x_new, n);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed, start, stop);
gpuTime += elapsed / 1000.0;
```

`TGPU` represents the **sum of timed GPU kernel durations**. It does **not** include input-file reading, memory allocation, CPU convergence checking, or host/device copies. The program appends measurements to `results_gpu_jacobi.csv` with the columns:

```csv
Run,N,Blocks,Threads,Iterations,TGPU
```

Each execution produces **12 data rows**: three matrix sizes multiplied by four configurations. The `Run` field increments across repeated program executions. Actual measured times are not included here because the CSV data was not provided.

### Verification and Comparison:

For each configuration, the recorded iteration count and kernel time can be compared to identify how the block/thread configuration affects Jacobi performance. To verify the numerical solution, the final vector should also be tested using the residual `||Ax - b||`, since the current stopping condition alone does not establish solution accuracy.

---

# Gauss–Seidel Iterative Method

Gauss–Seidel also solves **Ax = b** iteratively, but it differs from Jacobi in one important way: **as soon as a new solution component is calculated, it is used in subsequent component updates within the same iteration**. This often changes the convergence behavior, but it creates data dependencies that make straightforward CUDA parallelization more difficult.

> **Implementation note:** The two code blocks provided for this README are identical Jacobi CUDA implementations. No Gauss–Seidel source code or measured Seidel results were included. The explanation below describes the Gauss–Seidel algorithm accurately, but does not claim that particular kernels, thread settings, timing functions, or CSV outputs exist in the repository.

### Main Program Logic:

A typical Gauss–Seidel implementation reads the coefficient matrix, sets up the right-hand side and initial solution, and performs successive sweeps over the unknowns. During each sweep, it updates solution components **in order**, using newly calculated components as soon as they become available. After a sweep, it checks a stopping criterion and repeats if necessary.

### Reading Matrix:

The coefficient matrix can be stored in the same row-major layout as the Jacobi implementation:

```c
A[i * n + j]
```

The exact input filenames, right-hand-side initialization, and allocation strategy for the Gauss–Seidel program depend on its source code.

### Gauss–Seidel Logic:

The update equation is:

\[
x_i^{(k+1)} = \frac{b_i - \sum_{j=0}^{i-1} A_{ij}x_j^{(k+1)} - \sum_{j=i+1}^{n-1} A_{ij}x_j^{(k)}}{A_{ii}}
\]

The first summation uses the **new** values from the current sweep for indices `j < i`. The second uses the **old** values for indices `j > i`. In contrast, Jacobi uses old values for every term.

A conventional in-place CPU sweep resembles:

```c
for (int i = 0; i < n; i++) {
    double sum = b[i];
    for (int j = 0; j < n; j++) {
        if (j != i) sum -= A[i * n + j] * x[j];
    }
    x[i] = sum / A[i * n + i];
}
```

Because `x[i]` is overwritten immediately, later rows use its latest value.

### Parallel CPU and CUDA Considerations:

Unlike Jacobi, standard Gauss–Seidel cannot simply update all rows simultaneously from one old vector and still retain the standard Gauss–Seidel algorithm. Each row may depend on values updated by earlier rows in the same sweep. Possible parallel approaches include **multi-color/red–black ordering** for suitable matrices, block Gauss–Seidel, or other dependency-aware schedules. Their exact behavior depends on the matrix and implementation.

Therefore, the Jacobi CUDA grid-stride kernel shown earlier must **not** be labeled as a Gauss–Seidel kernel. The exact CUDA execution strategy can be documented once the Gauss–Seidel code is supplied.

### Convergence and Stopping Condition:

A common stopping rule compares the maximum change in successive iterates:

\[
\max_i |x_i^{(k+1)} - x_i^{(k)}| < \varepsilon
\]

The implementation's actual tolerance, residual checks, and maximum iteration count need to be confirmed from its source. As with Jacobi, convergence depends on matrix properties; a nonzero diagonal alone is not sufficient.

### Timing and Results:

For a fair comparison, Jacobi and Gauss–Seidel should be tested on the **same input matrices, right-hand sides, initial guesses, stopping tolerances, and hardware**. Record the number of iterations, kernel-only or CPU computation time, and—if available—end-to-end execution time. GPU kernel time and end-to-end time must not be treated as interchangeable measurements.

No Gauss–Seidel runtime figures or CSV column definitions have been supplied, so they are not fabricated here.

### Verification and Comparison:

The two methods differ mainly in when newly computed values become available:

| Property | Jacobi | Gauss–Seidel |
|---|---|---|
| Values used during a sweep | All from previous iteration | Latest values for already updated components |
| Independent row updates | Yes | Generally no |
| Straightforward row-wise CUDA parallelism | Yes | Requires special treatment of dependencies |
| Extra solution vector | Used by the provided implementation | Not necessary for a conventional in-place sweep |
| Guaranteed to converge for every matrix | No | No |

On some matrices, Gauss–Seidel may require fewer sweeps than Jacobi, but this is **not** an observed result for this assignment without measured data. Actual speed depends on convergence, the CPU/GPU implementation, synchronization, and the cost of each sweep.

### Overall Observations:

Jacobi is well suited to a row-wise CUDA kernel because each solution component can be calculated independently from the previous vector. Gauss–Seidel uses more recent values within each sweep, which changes its convergence behavior but introduces ordering dependencies. A meaningful assignment comparison should report actual iterations and runtimes along with residual-based correctness checks, rather than assuming one method is faster for all matrices.
