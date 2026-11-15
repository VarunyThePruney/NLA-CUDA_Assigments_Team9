#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

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

__global__ void jacobi(double *A, double *b, double *x, double *x_new, int n)
{
    int i = threadIdx.x;
    if (i < n)
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

    FILE *file = fopen("results_gpu.csv", "a");

    int run = 1;

    fseek(file, 0, SEEK_END);

    if (ftell(file) != 0)
    {
        FILE *readFile = fopen("results_gpu.csv", "r");

        char line[1024];
        int lastRun = 0;

        fgets(line, sizeof(line), readFile);

        while (fgets(line, sizeof(line), readFile))
        {
            int currentRun;

            if (sscanf(line, "%d,", &currentRun) == 1)
            {
                if (currentRun > lastRun)
                {
                    lastRun = currentRun;
                }
            }
        }

        run = lastRun + 1;

        fclose(readFile);
    }

    fseek(file, 0, SEEK_END);

    if (ftell(file) == 0)
    {
        fprintf(file, "Run,N,TGPU\n");
    }

    for (int s = 0; s < 3; s++)
    {
        cudaEvent_t start4, stop4, start8, stop8, start16, stop16, start32, stop32;
        cudaEventCreate(&start4);
        cudaEventCreate(&start8);
        cudaEventCreate(&start16);
        cudaEventCreate(&start32);
        cudaEventCreate(&stop4);
        cudaEventCreate(&stop8);
        cudaEventCreate(&stop16);
        cudaEventCreate(&stop32);

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
        }

        double *d_A, *d_b, *d_x, *d_x_new;
        cudaMalloc(&d_A, size * sizeof(double));
        cudaMalloc(&d_x, n * sizeof(double));
        cudaMalloc(&d_b, n * sizeof(double));
        cudaMalloc(&d_x_new, n * sizeof(double));
        cudaMemcpy(d_A, A, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

        while (1)
        {
            jacobi<<<4, 16>>>(d_A, d_b, d_x, n);
            cudaDeviceSynchronize();
            cudaMemcpy(d_x, d_x_new, n * sizeof(double), cudaMemcpyDeviceToHost);
        }
    }
}