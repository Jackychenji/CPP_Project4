#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include "Matrix.h"
#include <time.h>
#include <omp.h>
int main()
{

	size_t nowSize = 1000;
	Matrix *matrix1 = CreateMatrix(matrix1, nowSize, nowSize);
	Matrix *matrix2 = CreateMatrix(matrix2, nowSize, nowSize);
	float *value1 = (float *)malloc(sizeof(float) * nowSize * nowSize);
	float *value2 = (float *)malloc(sizeof(float) * nowSize * nowSize);
	InitialMatrix(matrix1, value1);
	InitialMatrix(matrix2, value2);
	long start, time1, time2, time3, result1, result2, result3;
	start = clock();
	printf("矩阵1 乘以 矩阵2: \n");
	Matrix *matrix3 = matmul_plain(matrix1, matrix2); //乘法
	// PrintMatrix(matrix3);
	// printf("\n");
	time1 = clock();
	Matrix *matrix4 = matmul_avx2(matrix1, matrix2);
	// PrintMatrix(matrix4);
	time2 = clock();
	Matrix *matrix5 = matmul_avx2_omp(matrix1, matrix2);
	// PrintMatrix(matrix5);
	time3 = clock();
	result1 = time1 - start;
	result2 = time2 - time1;
	result3 = time3 - time2;
	printf("plain运行时间:%f秒\n", (float)result1 / CLOCKS_PER_SEC);
	printf("SIMD运行时间:%f秒\n", (float)result2 / CLOCKS_PER_SEC);
	printf("OPENMP运行时间:%f秒\n", (float)result3 / CLOCKS_PER_SEC);

	return 0;
}