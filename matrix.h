#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
    size_t rows;
    size_t cols;
    float *entries;
} Matrix;

typedef float (*_activation_func)(float);

Matrix *matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix *m);
void matrix_fill(Matrix *m, float n);
void matrix_print(Matrix *m);
void matrix_copy(Matrix *dst, Matrix *src);
Matrix *matrix_row(Matrix *m, size_t row);
void matrix_randomize(Matrix *m, size_t n);
void matrix_apply(_activation_func func, Matrix *m);
void matrix_transpose(Matrix **m);

void matrix_dot(Matrix *dst, Matrix *a, Matrix *b);
void matrix_add(Matrix *dst, Matrix *m);
void matrix_scale(Matrix *dst, float n);
void matrix_subtract(Matrix *dst, Matrix *m);
void matrix_multiply(Matrix *dst, Matrix *a, Matrix *b);
#endif
