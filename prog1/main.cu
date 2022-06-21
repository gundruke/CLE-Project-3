#include "../common/common.h"
#include <stdio.h>
#include <unistd.h>
#include "matrix_processor.h"

//nvcc -O2 -Wno-deprecated-gpu-targets -include matrix_processor.c -o part1 main.cu

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

double *matrix, *cuda_matrix, *single_matrix;
double *results, *cuda_results, *cpu_results;
size_t bytes_read;

/** \brief The order of the matrix read*/
// number of matrices
int m = 0;
// matrix order
int n = 0;

int mem_size;
int mat_size;


static double get_delta_time(void);

__global__ void matrixProcessor(double *matrix, int matrix_order, double *results);

static void printUsage(char *cmdName);

int main(int argc, char **argv) {
    /* set up the device */
    int dev = 0;
    int i, j;
    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;

    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK (cudaSetDevice(dev));


    int opt; /* selected option */
    opterr = 0;

    do {
        switch ((opt = getopt(argc, argv, "f:n:h"))) {
            case 'f': /* file name */
                if (optarg[0] == '-') {
                    fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                    printUsage(basename(argv[0]));
                    return EXIT_FAILURE;
                }

                for (i = 2; i < argc; i++) {

                    (void) get_delta_time();
                    /* open the matrix and load the metadata*/
                    FILE *file;
                    printf("Currently on file : %s\n\n", basename(argv[i]));
                    file = fopen(argv[i], "rb");

                    if (!file) { /* Check that the file exists*/
                        printf("File does not exist \n");
                        exit(EXIT_FAILURE);
                    }

                    /*read the metadata from te file: number and order of matrices*/
                    bytes_read = fread(&m, sizeof(int), 1, file);
                    printf("Number of matrices to be read    : %d \n", m);
                    bytes_read = fread(&n, sizeof(int), 1, file);
                    printf("Matrices order                   : %d \n", n);

                    /* create the memory areas */
                    //size of each matrix
                    mat_size = n * n * sizeof(double);
                    //size of memory
                    mem_size = mat_size * m;

                    printf("Size of memory allocated         : %d bytes\n", mem_size);

                    matrix = (double *) malloc(mem_size);
                    CHECK(cudaMalloc((void **) &cuda_matrix, mem_size));
                    results = (double *) malloc(sizeof(double) * m);
                    cpu_results = (double *) malloc(sizeof(double) * m);
                    CHECK(cudaMalloc((void **) &cuda_results, sizeof(double) * m));

                    //TODO: proper define grid, not sure if its correct way
                    blockDimX = n;
                    blockDimY = 1;
                    blockDimZ = 1;
                    gridDimX = m;
                    gridDimY = 1;
                    gridDimZ = 1;

                    dim3 grid (gridDimX, gridDimY, gridDimZ);
                    dim3 block (blockDimX, blockDimY, blockDimZ);

                    printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> \n\n\n", gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);
                    printf("Initialization and opening of file took %.3e seconds\n", get_delta_time());

                    (void) get_delta_time();
                    read_matrix(matrix, file, n, m);
                    printf("Reading of all matrices took %.3e seconds\n", get_delta_time());


                    CHECK(cudaMemcpy(cuda_matrix, matrix, mem_size, cudaMemcpyHostToDevice));
                    printf("The transfer of %ld byte/s from the host to the device took %.3e seconds\n",
                           (long) mem_size, get_delta_time());


                    printf("\n\n###########   Processing on GPU   ###########\n");
                    matrixProcessor<<<grid, block>>>(cuda_matrix, n, cuda_results);
                    CHECK(cudaMemcpy(results, cuda_results, sizeof(double) * m, cudaMemcpyDeviceToHost));
                    printf("Processing of matrix on GPU took %.3e seconds\n\n", get_delta_time());


                    printf("\n###########   Processing on CPU   ###########\n");

                    for (j = 0; j < m; j++) {
                        single_matrix = (double *)malloc(mat_size);
                        memcpy(single_matrix, &matrix[n*n*j], mat_size);

                        cpu_results[j] = determinantCPU(n, single_matrix);
                        free(single_matrix);
                    }


                    printf("Processing of matrix on CPU took %.3e seconds\n\n", get_delta_time());

                    for (j = 0; j < m; j++) {
                        printf("Determinant for Matrix %d \t: (GPU) %e \t: (CPU) %e\n", j + 1, results[j], cpu_results[j]);
                    }

                    fclose(file);

                    CHECK(cudaFree(cuda_matrix));
                    CHECK(cudaFree(cuda_results));
                    // free host memory
                    free(matrix);
                    free(results);
                    free(cpu_results);

                    CHECK(cudaDeviceReset());

                }
                return 0;

                break;
            case 'h': /* help mode */
                printUsage(basename(argv[0]));
                return EXIT_SUCCESS;
            case '?': /* invalid option */
                fprintf(stderr, "%s: invalid option\n", basename(argv[0]));
                printUsage(basename(argv[0]));
                return EXIT_FAILURE;
            case -1:
                break;
        }
    } while (opt != -1);
    if (argc == 1) {
        fprintf(stderr, "%s: invalid format\n", basename(argv[0]));
        printUsage(basename(argv[0]));
        return EXIT_FAILURE;
    }
    exit(EXIT_SUCCESS);
}


static double get_delta_time(void) {
    static struct timespec t0, t1;

    t0 = t1;
    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (double) (t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double) (t1.tv_nsec - t0.tv_nsec);
}


__global__ void matrixProcessor(double *matrix, int matrix_order, double *results) {

    unsigned int tid, bid;
    bid = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.y;

    matrix += bid * matrix_order * matrix_order;


    int swapCount = 0;
    int i, k;
    double term = 0;

    for (i = 0; i < matrix_order - 1; i++) {
        //Partial Pivoting
        for (k = i + 1; k < matrix_order; k++) {
            //If diagonal element(absolute value) is smaller than any of the terms below it
            if (fabs(matrix[(i * matrix_order) + i]) < fabs(matrix[(k * matrix_order) + i])) {
                //Swap the rows
                swapCount++;

                double temp;
                temp = matrix[(i * matrix_order) + tid];
                matrix[(i * matrix_order) + tid] = matrix[(k * matrix_order) + tid];
                matrix[(k * matrix_order) + tid] = temp;
                __syncthreads();
            }

            __syncthreads();
            term = matrix[(k * matrix_order) + i] / matrix[(i * matrix_order) + i];
            matrix[(k * matrix_order) + tid] =
                    matrix[(k * matrix_order) + tid] - term * matrix[(i * matrix_order) + tid];

        }
    }

    results[bid] = determinant(matrix_order, matrix, swapCount);
}


static void printUsage(char *cmdName) {
    fprintf(stderr, "\nSynopsis: %s OPTIONS [filename]\n"
                    " OPTIONS:\n"
                    " -h --- print this help\n"
                    " -f --- filename\n",
            cmdName);
}