#include "../common/common.h"
#include <stdio.h>
#include "matrix_processor.h"

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

double *matrix, *cuda_matrix;
size_t bytes_read;

/** \brief The order of the matrix read*/
int matrix_order;
int matrix_id;
int num_matrices; /*stores the number of matrices in the file*/
int chunk_size;




__global__ void matrixProcessor(double* matrix, int matrix_order)
{

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
                temp = matrix[(i * matrix_order) + threadIdx.x];
                matrix[(i * matrix_order) + threadIdx.x] = matrix[(k * matrix_order) + threadIdx.x];
                matrix[(k * matrix_order) + threadIdx.x] = temp;
                __syncthreads();
            }

            term = matrix[(k * matrix_order) + i] / matrix[(i * matrix_order) + i];
            matrix[(k * matrix_order) +  threadIdx.x] = matrix[(k * matrix_order) + threadIdx.x] - term * matrix[(i * matrix_order) +  threadIdx.x];
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        printf("Hello World from GPU %d %e! \n", threadIdx.x, determinant(matrix_order, matrix, swapCount));
    }
}



int main(int argc, char **argv)
{


    FILE *file;
    file = fopen("mat128_32.bin", "rb");

    if (!file)
    { /* Check that the file exists*/
        printf("File does not exist \n");
        exit(EXIT_FAILURE);
    }

    /*display statistics about the current file*/
    /*read the metadata from te file: number and order of matrices*/
    bytes_read = fread(&num_matrices, sizeof(int), 1, file);
    printf("Number of matrices to be read    : %d \n", num_matrices);
    bytes_read = fread(&matrix_order, sizeof(int), 1, file);
    printf("Matrices order                   : %d \n", matrix_order);

    chunk_size = matrix_order * matrix_order;
    matrix = (double *)malloc((chunk_size) * sizeof(double));


    read_matrix(matrix, file, matrix_order);
    printf("here... \n");
    cudaMalloc((void **)&cuda_matrix, (chunk_size) * sizeof(double));
    cudaMemcpy(cuda_matrix, matrix, (chunk_size ) * sizeof(double), cudaMemcpyHostToDevice);

    matrixProcessor<<<1, matrix_order>>>(cuda_matrix, matrix_order);

    fclose(file);



//    printf("Hello World from CPU %e!\n", determinant(matrix_order, matrix));

//    helloFromGPU<<<1, 1>>>(cuda_matrix, matrix_order, swapCount);
    CHECK(cudaDeviceReset());
    return 0;
}


