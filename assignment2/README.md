# Cholesky Factorization

SE24UCSE242 - Varun Sai Golakoti  
SE24UCSE244 - Konakalla Sai Shloka

---

https://github.com/VarunyThePruney/NLA-CUDA_Assigment1/tree/main/assignment2

# Cholesky Factorization:

Cholesky factorization consists of 2 main loops:

1. **Sizes Loop:** Iterates 3 times for 1024x1024, 2048x2048 and 4096x4096 matrices. It reads the matrix from the input files using readMatrix() and allocates memory for the resulting lower triangular matrix l. This loop also handles the final output of the best runtime for each matrix.

2. **Threads Loop:** Iterates 4 times for 2, 4, 8 and 16 threads. For each thread option, a new lower triangular matrix l is initialized using calloc(). Then, omp_set_num_threads() sets the number of threads to be used. The cholesky() function is called and its runtime is recorded as Tp. After all four thread options finish, the lowest runtime is found and stored as TCPU_best. Finally, the results are written to results_parallel.csv.

<h3>Reading Matrix:</h3>

The function readMatrix() reads the matrix order N from the first line of the input file. It then allocates memory for an N x N matrix and reads all the remaining values into the array a.

Since we are using row-major storage, the element at row i and column j is accessed using:

&emsp;&emsp;&emsp;&emsp;a[i * n + j]

<h3>Cholesky Logic:</h3>

Cholesky factorization decomposes a symmetric positive definite matrix A into two matrices:

&emsp;&emsp;&emsp;&emsp;A = LL(Transpose)

Where L is a lower triangular matrix and L(T) is its transpose.

The main logic resides in the cholesky() function. It starts with an outer loop k which iterates through every column of the matrix. For each column, the diagonal element is calculated first.

The diagonal calculation is:

&emsp;&emsp;&emsp;&emsp;L[k][k] = √(A[k][k] - ∑ L[k][j]²)

The inner loop j calculates the summation using previously calculated elements in the same row. Once the summation is subtracted from the original diagonal element, its square root is stored in l[k * n + k].

Next, all elements below the diagonal in the current column are calculated using:

&emsp;&emsp;&emsp;&emsp;L[i][k] = (A[i][k] - ∑ L[i][j] \* L[k][j]) / L[k][k]

Here, i represents each row below the current diagonal, while j calculates the summation using previously computed elements.

<h3>Parallel CPU Logic:</h3>

The main difference between the sequential and parallel versions is how the rows below the diagonal are processed.

Instead of using a normal loop to calculate each row one after another, we use multithreading:

&emsp;&emsp;&emsp;&emsp;#pragma omp parallel for

This is placed directly before the loop that iterates through i, from k + 1 to n - 1.

OpenMP automatically distributes these loop iterations among the available threads. Each thread calculates a different row in the current column, meaning multiple elements can be calculated simultaneously.

The inner j loop remains sequential because every element requires the summation of previously calculated values.

Once all threads finish calculating the current column, and automatically waits for them before continuing to the next iteration of k. This is important because the next column depends on the values calculated in the previous column.

The number of threads is controlled using:

&emsp;&emsp;&emsp;&emsp;omp_set_num_threads(threads[t]);

This allows us to test the same factorization using 2, 4, 8 and 16 threads without changing the actual cholesky logic.

<h3>Timing and Results:</h3>

We wrote most of this code before Praveen sir discussed cudaEventTimers in class, so we used gettimeofday() to track the times needed as per the assignemnt.

Before calling cholesky(), the starting time is recorded. After the function finishes, the elapsed time is calculated and stored in runtime[t].

The best runtime is found by comparing all four thread configurations:

&emsp;&emsp;&emsp;&emsp;best = minimum(T2, T4, T8, T16)

This value is stored as TCPU_best.

Finally, the program writes the matrix name, matrix order, thread count, individual runtime (Tp) and best runtime (TCPU_best) into results_parallel.csv.

The CSV contains 12 rows in total, with 4 thread configurations for each of the 3 matrix sizes.

<h3>Verification and Comparison:</h3>
The results show that the parallel CPU implementation performs better than the sequential CPU implementation across all three matrix sizes. The best parallel CPU performance is consistently achieved using 2 threads, with average execution times of 0.4470 seconds, 3.7734 seconds and 29.1964 seconds for matrix sizes 1024x1024, 2048x2048 and 4096x4096, respectively. Compared to the sequential CPU averages of 0.5975 seconds, 4.8585 seconds and 39.0045 seconds, this provides a modest improvement in execution time.

One thing I did notice that I did not expect is that increasing threads increased the time for parallel CPU. This might be a mistake in how I built the code but my guess is that the synchronization for cholesky has high overhead.

The GPU implementation achieves significantly lower execution times, with averages of 0.1143 seconds, 0.4438 seconds and 2.0462 seconds for the three matrix sizes. However, the T(end to end) the averages increase to 0.4980 seconds, 0.9407 seconds and 4.0427 seconds. For the 1024x1024 matrix, the end-to-end GPU time is only slightly better than the best parallel CPU time. For the 2048x2048 and 4096x4096 matrices, the GPU provides a much greater improvement since it has faster working speed.

<h3>Comparing Cholesky to Gaussian method</h3>
Comparing the two, they have around equal results for the 2048x2048, with gaussian being around 1.36x faster. Which doesn't make sense as Cholesky requires less arithmatic operations due to the symmetric input. This may just be due to the implementation or some other external factor im missing. But by theory, gaussian elimination should be slower than cholesky, but by output, gaussian was faster.
