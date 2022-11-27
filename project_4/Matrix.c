#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include "Matrix.h"

Matrix *CreateMatrix(Matrix *matrix, size_t row, size_t column)
{
    if (row > 0 && column > 0)
    {
        matrix = (Matrix *)malloc(sizeof(Matrix));
        matrix->row = row;
        matrix->column = column;
        matrix->size = row * column;
        matrix->data = (float *)malloc(sizeof(float) * row * column);
        memset(matrix->data, 0, sizeof(float) * row * column);
        return matrix;
    }
    else
        return NULL;
}

void InitialMatrix(Matrix *matrix, float *value)
{
    for (size_t i = 0; i < matrix->size; i++)
    {
        value[i] = (float)0 + 1.0 * rand() / RAND_MAX * 10;
    }
    if (matrix->data != NULL)
    {
        memcpy(matrix->data, value, matrix->row * matrix->column * sizeof(float));
    }
}

void FreeMatrix(Matrix *matrix)
{
    free(matrix->data);
    matrix->data = NULL;
    free(matrix);
    matrix = NULL;
}

void PrintMatrix(Matrix *matrix)
{
    for (size_t i = 0; i < matrix->size; i++)
    {
        printf("%lf\t", matrix->data[i]);
        if ((i + 1) % matrix->column == 0)
            printf("\n");
    }
}

void CopyMatrix(Matrix *A, Matrix *B)
{
    B->row = A->row;
    B->column = A->column;
    B->size = A->size;
    memcpy(B->data, A->data, A->size * sizeof(float));
}

void TransMatrix(Matrix *matrix)
{
    Matrix *matrixTemp = CreateMatrix(matrixTemp, matrix->row, matrix->column);
    CopyMatrix(matrix, matrixTemp);
    for (int i = 0; i < matrix->row; i++)
    {
        for (int j = 0; j < matrix->column; j++)
        {
            matrix->data[i * matrix->row + j] = matrixTemp->data[j * matrix->row + i];
        }
    }
    FreeMatrix(matrixTemp);
}

Matrix *matmul_plain(Matrix *A, Matrix *B)
{
    if (B->row == A->column)
    {
        Matrix *C = CreateMatrix(C, A->row, B->column);
        TransMatrix(B);
        for (size_t i = 0; i < A->size; i += A->row)
        {
            
            for (size_t j = 0; j < B->size; j += B->row)
            {
                for (size_t k = 0; k < A->row; k++)
                {
                    C->data[i + j / B->row] += A->data[i + k] * B->data[j + k];
                }
            }
        }
        TransMatrix(B);
        return C;
    }
    else
    {
        printf("Invalid Input\n");
        return NULL;
    }
}

Matrix *matmul_avx2(Matrix *A, Matrix *B)
{
    if (A->column == B->row)
    {
        Matrix *C = CreateMatrix(C, A->row, B->column);
        float *p1 = (float *)(aligned_alloc(256, 256 * A->size * sizeof(float)));
        float *p2 = (float *)(aligned_alloc(256, 256 * B->size * sizeof(float)));
        memcpy(p1, A->data, A->size * sizeof(float));
        TransMatrix(B);
        memcpy(p2, B->data, B->size * sizeof(float));
        float sum[8] = {0};
        __m256 a, b;
        __m256 c = _mm256_setzero_ps();

        for (size_t i = 0; i < A->size; i += A->row)
        {
            for (size_t j = 0; j < B->size; j += B->row)
            {
                for (size_t k = 0; k < A->row; k += 8)
                {

                    a = _mm256_load_ps(&p1[i] + k);
                    b = _mm256_load_ps(&p2[j] + k);
                    c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
                }
                _mm256_store_ps(sum, c);
                float total = (sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]);
                C->data[i + j / B->row] = total;
            }
        }
        return C;
    }
    else
    {
        printf("Invalid Input\n");
        return NULL;
    }
}

Matrix *matmul_omp(Matrix *A, Matrix *B)
{
    if (B->row == A->column)
    {

        Matrix *C = CreateMatrix(C, A->row, B->column);
        TransMatrix(B);
#pragma omp parallel for num_threads(4)
        for (size_t i = 0; i < A->size; i += A->row)
        {
            for (size_t j = 0; j < B->size; j += B->row)
            {
                for (size_t k = 0; k < A->row; k++)
                {
                    C->data[i + j / B->row] += A->data[i + k] * B->data[j + k];
                }
            }
        }
        TransMatrix(B);
        return C;
    }
    else
    {
        printf("Invalid Input\n");
        return NULL;
    }
}

Matrix *matmul_avx2_omp(Matrix *A, Matrix *B)
{
    if (A->column == B->row)
    {
        Matrix *C = CreateMatrix(C, A->row, B->column);
        float *p1 = (float *)(aligned_alloc(256, 256 * A->size * sizeof(float)));
        float *p2 = (float *)(aligned_alloc(256, 256 * B->size * sizeof(float)));
        memcpy(p1, A->data, A->size * sizeof(float));
        TransMatrix(B);
        memcpy(p2, B->data, B->size * sizeof(float));
        float sum[8] = {0};
        __m256 a, b;
        __m256 c = _mm256_setzero_ps();
	omp_set_num_threads(4);
#pragma omp parallel for
        for (size_t i = 0; i < A->size; i += A->row)
        {
            for (size_t j = 0; j < B->size; j += B->row)
            {
#pragma omp critical
                for (size_t k = 0; k < A->row; k += 8)
                {
                    a = _mm256_load_ps(&p1[i] + k);
                    b = _mm256_load_ps(&p2[j] + k);
                    c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
                }

                _mm256_store_ps(sum, c);
                float total = (sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]);
                C->data[i + j / B->row] = total;
            }
        }
        return C;
    }
    else
    {
        printf("Invalid Input\n");
        return NULL;
    }
}
