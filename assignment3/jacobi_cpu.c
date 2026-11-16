
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

double getTime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1000000.0;
}

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

void jacobi(double *A, double *b, double *x,
            double *x_new, int n)
{
    for (int i = 0; i < n; i++)
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

    FILE *file = fopen("results_cpu_jacobi.csv", "a+");

    int run = 1;
    char line[1024];

    rewind(file);

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
        fprintf(file, "Run,N,Iterations,TCPU(s)\n");
    }

    for (int s = 0; s < 3; s++)
    {
        double *A;
        int n;

        readMatrix(filenames[s], &A, &n);

        double *b = (double *)malloc(n * sizeof(double));
        double *x = (double *)malloc(n * sizeof(double));
        double *x_new = (double *)malloc(n * sizeof(double));

        for (int i = 0; i < n; i++)
        {
            b[i] = 1;
            x[i] = 0;
        }

        int iterations = 0;
        double maxDiff;
        double start = getTime();
        while (1)
        {
            jacobi(A, b, x, x_new, n);
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

            if (maxDiff < 1e-6)
            {
                break;
            }
        }
        double end = getTime();
        double cpuTime = end - start;

        printf("N = %d\n", n);
        printf("Iterations = %d\n", iterations);
        printf("TCPU = %.6f seconds\n\n", cpuTime);
        fprintf(file, "%d,%d,%d,%.6f\n", run, n, iterations, cpuTime);

        fflush(file);

        free(A);
        free(b);
        free(x);
        free(x_new);
    }
    fclose(file);
}
