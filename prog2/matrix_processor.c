/**
 *  \file word_counter.c (implementation file)
 *
 *  \brief Problem name: Text Processing
 *
 * implementation of the algorithm used for processing the different words
 *
 *  \author Richard Jonker and Roshan Poudel
 */


#include <stdio.h>
#include <string.h>
#include <math.h>

extern size_t bytes_read;


/**
 *  \brief Extracts a single UTF8 character from a given FILE buffer.
 *
 *  Its role is to take a variable length array of bytes that represent a single character from a file.
 *
 *  \param buffer The FILE buffer where the UTF-8 characters are stored
 *  \param utf8char The pointer to the char array where the UTF-8 character will be stored
 */

// CPU Code
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
            if (fabs(mat[(i * mat_order) + i]) < fabs(mat[(i * mat_order) + k]))
            {
                // Swap the columns
                swapCount++;
                for (j = 0; j < mat_order; j++)
                {
                    double temp;
                    temp = mat[(j * mat_order) + i];
                    mat[(j * mat_order) + i] = mat[(j * mat_order) + k];
                    mat[(j * mat_order) + k] = temp;
                }
            }
        }
        // Begin Gauss Elimination
        for (k = i + 1; k < mat_order; k++)
        {
            double term = mat[(i * mat_order) + k] / mat[(i * mat_order) + i];
            for (j = 0; j < mat_order; j++)
            {
                mat[(j * mat_order) + k] = mat[(j * mat_order) + k] - term * mat[(j * mat_order) + i];
            }
        }
    }
    return swapCount;
}


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




// GPU Code
__device__ double determinant(int mat_order, double *mat, int swapCount) {
    double det = 1;
    int i;

    for (i = 0; i < mat_order; i++) {
        det = det * mat[(i * mat_order) + i];
    }
    return det * pow(-1, swapCount);
}

// Common
void read_matrix(double *mat, FILE *file, int mat_order, int mat_size) {
    for (int j = 0; j < (mat_order * mat_order*mat_size); ++j) {
        bytes_read = fread(&mat[j], sizeof(double), 1, file);
    }
}

