
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <sys/time.h>

void readMatrix(char *filename, double **a, int *n)
{
    FILE *file = fopen(filename, "r");
    fscanf(file, "%d", n);

    int size = (*n) * (*n);
    *a = (double *)malloc(size * sizeof(double));

    for (int i = 0; i < size; i++)
    {
        fscanf(file, "%lf", &(*a)[i]);
    }

    fclose(file);
}

double getTime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000.0 + t.tv_usec / 1000.0;
}

__global__ void jacobi(double *A, double *b, double *x,
                       double *x_new, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride)
    {
        double sum = b[i];

        for (int j = 0; j < n; j++)
        {
            if (j != i)
            {
                sum -= A[i * n + j] * x[j];
            }
        }

        x_new[i] = sum / A[i * n + i];
    }
}

int main()
{
    char *filenames[] = {"Cholesky_Input_1024x1024.txt", "Cholesky_Input_2048x2048.txt", "Cholesky_Input_4096x4096.txt"};

    int blocks[] = {4, 8, 16, 32};
    int threads[] = {32, 64, 128, 256};

    FILE *file = fopen("results_jacobi_gpu.csv", "a+");

    int run = 1;

    rewind(file);

    char line[1024];

    while (fgets(line, sizeof(line), file))
    {
        int currentRun;

        if (sscanf(line, "%d,", &currentRun) == 1)
        {
            if (currentRun >= run)
            {
                run = currentRun + 1;
            }
        }
    }

    fseek(file, 0, SEEK_END);

    if (ftell(file) == 0)
    {
        fprintf(file, "Run,N,Blocks,Threads,Iterations,TGPU,Te2e\n");
    }

    for (int s = 0; s < 3; s++)
    {
        double *A;
        int n;

        readMatrix(filenames[s], &A, &n);

        int size = n * n;

        double *b = (double *)malloc(n * sizeof(double));
        double *x = (double *)malloc(n * sizeof(double));
        double *x_new = (double *)malloc(n * sizeof(double));

        for (int i = 0; i < n; i++)
        {
            b[i] = 1;
            x[i] = 0;
        }

        double *d_A, *d_b, *d_x, *d_x_new;

        cudaMalloc(&d_A, size * sizeof(double));
        cudaMalloc(&d_b, n * sizeof(double));
        cudaMalloc(&d_x, n * sizeof(double));
        cudaMalloc(&d_x_new, n * sizeof(double));

        cudaMemcpy(d_A, A, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);

        for (int c = 0; c < 4; c++)
        {
            int B = blocks[c];
            int T = threads[c];

            cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            int iterations = 0;
            double maxDiff;
            double gpuTime = 0;

            double startTime = getTime();

            while (1)
            {
                cudaEventRecord(start);

                jacobi<<<B, T>>>(d_A, d_b, d_x, d_x_new, n);

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float elapsed;
                cudaEventElapsedTime(&elapsed, start, stop);

                gpuTime += elapsed;

                cudaMemcpy(x_new, d_x_new, n * sizeof(double), cudaMemcpyDeviceToHost);

                maxDiff = 0;

                for (int i = 0; i < n; i++)
                {
                    double diff = fabs(x_new[i] - x[i]);

                    if (diff > maxDiff)
                    {
                        maxDiff = diff;
                    }
                }

                iterations++;

                double *temp = x;
                x = x_new;
                x_new = temp;

                double *d_temp = d_x;
                d_x = d_x_new;
                d_x_new = d_temp;

                if (maxDiff < 1e-6)
                {
                    break;
                }
            }

            double endTime = getTime();
            double totalTime = endTime - startTime;

            printf("N = %d, Blocks = %d, Threads = %d\n", n, B, T);
            printf("Iterations = %d\n", iterations);
            printf("GPU Time = %.4f ms\n", gpuTime);
            printf("Total Time = %.4f ms\n\n", totalTime);

            fprintf(file, "%d,%d,%d,%d,%d,%.4f,%.4f\n",
                    run, n, B, T, iterations, gpuTime, totalTime);

            fflush(file);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            for (int i = 0; i < n; i++)
            {
                x[i] = 0;
            }
        }

        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_x_new);

        free(A);
        free(b);
        free(x);
        free(x_new);
    }

    fclose(file);

    return 0;
}
