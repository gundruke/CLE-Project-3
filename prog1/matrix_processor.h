/**
 *  \file word_counter.c (interface file)
 *
 *  \brief Problem name: Text Processing
 *
 * implementation of the algorithm used for processing the different words
 *
 *  \author Richard Jonker and Roshan Poudel
 */

#include <stdio.h>

#ifndef MATRIX_PROCESSOR_H
#define MATRIX_PROCESSOR_H

__device__ void gaussElimination(int mat_order, double *mat);

__device__ double determinant(int mat_order, double *mat);

extern void read_matrix(double *mat, FILE *file, int mat_order);
#endif //MATRIX_PROCESSOR_H
