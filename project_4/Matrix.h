#pragma once
#include <stddef.h>
typedef struct
{
	size_t row, column, size; 
	float *data;
} Matrix;

Matrix *CreateMatrix(Matrix *matrix, size_t row, size_t column);
void InitialMatrix(Matrix *matrix, float *value);				 
void FreeMatrix(Matrix *matrix);							
void PrintMatrix(Matrix *matrix); 
void TransMatrix(Matrix *matrix);
Matrix *matmul_plain(Matrix *matrix_A, Matrix *matrix_B);
Matrix *matmul_avx2(Matrix *A, Matrix *B);
Matrix *matmul_avx2_omp(Matrix *A, Matrix *B);
Matrix *matmul_omp(Matrix *A, Matrix *B);

