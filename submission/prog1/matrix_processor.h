/**
 *  \file matrix_processing.c (implementation file)
 *
 *  \brief Problem name: Calculating determinant by Gaussian Elimination
 *
 * implementation of the algorithm used for calculating determinant with gaussian elimination (Row Based approach)
 *
 *  \author Richard Jonker and Roshan Poudel
 */


#include <stdio.h>

#ifndef MATRIX_PROCESSOR_H
#define MATRIX_PROCESSOR_H

int gaussEliminationCPU(int mat_order, double *mat);
double determinantCPU(int mat_order, double *mat);

__device__ double determinant(int mat_order, double *mat, int swapCount);
extern void read_matrix(double *mat, FILE *file, int mat_order, int mat_size);
#endif //MATRIX_PROCESSOR_H
