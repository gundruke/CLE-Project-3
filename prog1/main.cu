#include "../common/common.h"
#include <stdio.h>
#include "matrix_processor.h"

//nvcc -O2 -Wno-deprecated-gpu-targets -include matrix_processor.c -o part1 main.cu

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

double *matrix, *cuda_matrix;
double *results, *cuda_results;
size_t bytes_read;

/** \brief The order of the matrix read*/
// number of matrices
int m = 0;
// matrix order
int n = 0;




// todo add
//static double get_delta_time(void)
//{
//    static struct timespec t0,t1;
//
//    t0 = t1;
//    if(clock_gettime(CLOCK_MONOTONIC,&t1) != 0)
//    {
//        perror("clock_gettime");
//        exit(1);
//    }
//    return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
//}




__global__ void matrixProcessor(double *matrix, int matrix_order, double * results) {

    unsigned int tid, bid;
    bid = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.y;

    matrix += bid*matrix_order*matrix_order;


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


int main(int argc, char **argv) {
    /* set up the device */
    int dev = 0;

    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK (cudaSetDevice(dev));

    //todo load the data properly (cli)

    /* open the matrix and load the metadata*/
    FILE *file;
    file = fopen("mat128_32.bin", "rb");

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
    int mat_size = n * n * sizeof(double);
    //size of memory
    int mem_size = mat_size * m;

    printf("Size of memory allocated         : %d bytes\n", mem_size);

    matrix = (double *) malloc(mem_size);
    CHECK(cudaMalloc((void **) &cuda_matrix, mem_size));
    results = (double *) malloc(sizeof(double) * m);
    CHECK(cudaMalloc((void **) &cuda_results, sizeof(double) * m));

//    (void) get_delta_time ();
    read_matrix(matrix, file, n, m);
//    printf ("The initialization of matrices %.3e seconds\n", get_delta_time ());


    CHECK(cudaMemcpy(cuda_matrix, matrix, mem_size, cudaMemcpyHostToDevice));
//    printf ("The transfer of %ld byte/s from the host to the device took %.3e seconds\n",
//            (long) mem_size, get_delta_time ());


    //todo proper groper grid
    matrixProcessor<<<m, n>>>(cuda_matrix, n, cuda_results);

    CHECK(cudaMemcpy(results, cuda_results, sizeof(double) * m, cudaMemcpyDeviceToHost));

    for (int i = 0; i<m; i++) {
        printf("Determinant for Matrix %d : %e\n", i+1, results[i]);
    }

    fclose(file);

    CHECK(cudaFree(cuda_matrix));
    CHECK(cudaFree(cuda_results));
    // free host memory
    free(matrix);
    free(results);
    CHECK(cudaDeviceReset());
    return 0;
}






