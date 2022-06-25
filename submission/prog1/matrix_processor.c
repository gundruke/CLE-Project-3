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
#include <string.h>
#include <math.h>

extern size_t bytes_read;


/**
 *  \brief Performs Gaussian Elimination - CPU code
 *
 *  \param mat_order Matrix order
 *  \param mat Matrix pointer
 */
int gaussEliminationCPU(int mat_order, double *mat)
{
    int i, j, k;
    int swapCount = 0;
    for (i = 0; i < mat_order - 1; i++)
    {
        // Partial Pivoting
        for (k = i + 1; k < mat_order; k++)
        {
            // If diagonal element(absolute value) is smaller than any of the terms below it
            if (fabs(mat[(i * mat_order) + i]) < fabs(mat[(k * mat_order) + i]))
            {
                // Swap the rows
                swapCount++;
                for (j = 0; j < mat_order; j++)
                {
                    double temp;
                    temp = mat[(i * mat_order) + j];
                    mat[(i * mat_order) + j] = mat[(k * mat_order) + j];
                    mat[(k * mat_order) + j] = temp;
                }
            }
        }
        // Begin Gauss Elimination
        for (k = i + 1; k < mat_order; k++)
        {
            double term = mat[(k * mat_order) + i] / mat[(i * mat_order) + i];
            for (j = 0; j < mat_order; j++)
            {
                mat[(k * mat_order) + j] = mat[(k * mat_order) + j] - term * mat[(i * mat_order) + j];
            }
        }
    }
    return swapCount;
}


/**
 *  \brief Performs Determinant calculation - CPU code
 *
 *  \param mat_order Matrix order
 *  \param mat Matrix pointer
 */
double determinantCPU(int mat_order, double *mat)
{
    double det = 1;
    int i;
    int swapCount = gaussEliminationCPU(mat_order, mat);
    for (i = 0; i < mat_order; i++)
    {
        det = det * mat[(i * mat_order) + i];
    }
    return det * pow(-1, swapCount);
}




/**
 *  \brief Calculate determinant using GPU
 *
 *  \param mat_order Matrix order (Number of rows/Columns)
 *  \param swapCount Number of swaps performed during Gaussian Elimination
 */
__device__ double determinant(int mat_order, double *mat, int swapCount) {
    double det = 1;
    int i;

    for (i = 0; i < mat_order; i++) {
        det = det * mat[(i * mat_order) + i];
    }
    return det * pow(-1, swapCount);
}


/**
 *  \brief Read the matrix from the file
 *
 *  \param mat Pointer to store matrix
 *  \param file File pointer
 *  \param mat_order Matrix order (Number of rows/Columns)
 *  \param mat_size Total number of matrices to read
 */
void read_matrix(double *mat, FILE *file, int mat_order, int mat_size) {
    for (int j = 0; j < (mat_order * mat_order*mat_size); ++j) {
        bytes_read = fread(&mat[j], sizeof(double), 1, file);
    }
}
