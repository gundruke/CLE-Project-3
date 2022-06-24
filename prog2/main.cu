#include "../common/common.h"
#include <stdio.h>
#include <unistd.h>
#include "matrix_processor.h"

//To compiler
//nvcc -O2 -Wno-deprecated-gpu-targets -include matrix_processor.c -o part2 main.cu

/*
 * Calculates determinant using CUDA and CPU then print them once calculated.
 * CUDA program to calculate determinant using Gaussian Elimination method
 * Used the column swapping approach
 */

double *matrix, *cuda_matrix, *single_matrix;
double *results, *cuda_results, *cpu_results;
size_t bytes_read;

/** \brief The order of the matrix read*/
int m = 0; // number of matrices
int n = 0; // order of matrices

int mem_size; // size of memory for matrices of one file
int mat_size; // size of memory for one matrix


/**
 *  \brief Calculate the time difference
 *
 *  \param mat_order Matrix order (Number of rows/Columns)
 *  \param swapCount Number of swaps performed during Gaussian Elimination
 */
static double get_delta_time(void);


/**
 *  \brief Process the matrix on GPU and stores the determinant on results pointer
 *
 *  \param matrix Matrix pointer
 *  \param matrix_order Order of the matrix passed
 *  \param results To store the results (determinant) of matrices
 */
__global__ void matrixProcessor(double *matrix, int matrix_order, double *results);


/**
 *  \brief Calculates the cosine similarity between the passed vectors
 *
 *  \param A first vector
 *  \param B second vector
 *  \param Vector_Length length of the vectors passed
 */
double cosine_similarity(double *A, double *B, unsigned int Vector_Length);


/**
 *  \brief Print the Help usage when help flag is passed or wrong flags are passed
 *
 *  \param cmdName command passed
 */
static void printUsage(char *cmdName);

// main function
int main(int argc, char **argv) {
    /* set up the device */
    int dev = 0;
    int i, j;

    // Grid and Block dimension variables
    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;

    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK (cudaSetDevice(dev));


    int opt; /* selected option */
    opterr = 0;

    do {
        // Parse the flags passed in the command line
        switch ((opt = getopt(argc, argv, "f:n:h"))) {
            case 'f': /* file name */
                if (optarg[0] == '-') {
                    fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                    printUsage(basename(argv[0]));
                    return EXIT_FAILURE;
                }
                // Iterate through filenames passed
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

                    // Block dimensions
                    blockDimX = n; // matrix order (Each block thread processes a row)
                    blockDimY = 1;
                    blockDimZ = 1;
                    // define block dimension
                    dim3 block (blockDimX, blockDimY, blockDimZ);

                    gridDimX = m; // number of matrices (Each block processes a matrix)
                    gridDimY = 1;
                    gridDimZ = 1;
                    // define grid dimension
                    dim3 grid (gridDimX, gridDimY, gridDimZ);

                    printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> \n\n\n", gridDimX, gridDimY, gridDimZ,
                           blockDimX, blockDimY, blockDimZ);
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
                        single_matrix = (double *) malloc(mat_size);
                        memcpy(single_matrix, &matrix[n * n * j], mat_size);

                        cpu_results[j] = determinantCPU(n, single_matrix);
                        free(single_matrix);
                    }


                    printf("Processing of matrix on CPU took %.3e seconds\n\n", get_delta_time());

                    for (j = 0; j < m; j++) {
                        printf("Determinant for Matrix %d \t: (GPU) %e \t: (CPU) %e \n", j + 1, results[j],
                               cpu_results[j]);
                    }

                    //printf("%lf\n", cosine_similarity(results, cpu_results, m));
                    if ((float)cosine_similarity(results, cpu_results, m) >= 0.99999999999999999999999)
                        printf("\nGPU and CPU determinants are the same.\n\n");
                    else
                        printf("\nGPU and CPU determinants are different.\n\n");


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

            if (fabs(matrix[(i * matrix_order) + i]) < fabs(matrix[(i * matrix_order) + k])) {
                //Swap the rows
                swapCount++;
                __syncthreads();

                double temp;
                temp = matrix[(tid * matrix_order) + i];
                matrix[(tid * matrix_order) + i] = matrix[(tid * matrix_order) + k];
                matrix[(tid * matrix_order) + k] = temp;
            }

            __syncthreads();
            term = matrix[(i * matrix_order) + k] / matrix[(i * matrix_order) + i];
            __syncthreads();
            matrix[(tid * matrix_order) + k] =
                    matrix[(tid * matrix_order) + k] - term * matrix[(tid * matrix_order) + i];
            __syncthreads();

        }
    }
    //__syncthreads();

    results[bid] = determinant(matrix_order, matrix, swapCount);
}


double cosine_similarity(double *A, double *B, unsigned int Vector_Length) {
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (unsigned int i = 0u; i < Vector_Length; ++i) {
        dot += A[i] * B[i];
        denom_a += A[i] * A[i];
        denom_b += B[i] * B[i];
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}


static void printUsage(char *cmdName) {
    fprintf(stderr, "\nSynopsis: %s OPTIONS [filename]\n"
                    " OPTIONS:\n"
                    " -h --- print this help\n"
                    " -f --- filename\n",
            cmdName);
}
