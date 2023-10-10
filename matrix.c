#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <math.h>

static float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

// User should free the matrix
Matrix *matrix_create(size_t rows, size_t cols) {
    assert(rows > 0);
    assert(cols > 0);
    Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
    assert(matrix != NULL);
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->entries = malloc(rows * cols * sizeof(float));
    assert(matrix->entries != NULL);
    return matrix;
}

void matrix_free(Matrix *m) {
    free(m->entries);
    free(m);
}

void matrix_fill(Matrix *m, float n) {
    assert(m != NULL);
    for (size_t i = 0; i < m->rows; ++i) {
        for (size_t j = 0; j < m->cols; ++j) {
            m->entries[i * m->cols + j] = n;
        }
    }
}

void matrix_print(Matrix *m) {
    size_t padding = 4;
    assert(m != NULL);
    printf("Rows: %d, cols: %d\n", m->rows, m->cols);
    for (size_t i = 0; i < m->rows; ++i) {
        printf("%*s    ", (int)padding, "");
        for (size_t j = 0; j < m->cols; ++j) {
            printf("%f ", m->entries[i * m->cols + j]);
        }
        printf("\n");
    }
}

void matrix_copy(Matrix *dst, Matrix *src) {
    assert(dst->cols == src->cols);
    assert(dst->rows == src->rows);
    for (size_t i = 0; i < dst->rows; ++i) {
        for (size_t j = 0; j < dst->cols; ++j) {
            dst->entries[i * dst->cols + j] = src->entries[i * src->cols + j];
        }
    }
}

void matrix_rotate_row(Matrix **m) {
    assert((*m)->rows == 1);
    Matrix *temp = matrix_create((*m)->cols, 1);
    for (size_t i = 0; i < (*m)->cols; ++i) {
        temp->entries[i] = (*m)->entries[i];
    }
    matrix_free(*m);
    *m = temp;
}

// User should free the matrix
Matrix *matrix_row(Matrix *m, size_t row) {
    assert(m != NULL);
    assert(row >= 0 && row < m->rows);
    Matrix *row_m = matrix_create(1, m->cols);
    for (size_t i = 0; i < m->cols; ++i) {
        row_m->entries[i] = m->entries[row * m->cols + i];
    }

    return row_m;
}

float uniform_distribution(float low, float high) {
    float difference = high - low;
    size_t scale = 10000;
    size_t scaled_difference = (size_t)(difference * scale);
    return low + (1.0 * (rand() % scaled_difference) / scale);
}

void matrix_randomize(Matrix *m, size_t n) {
    float min = -1.0 / sqrt(n);
    float max = 1.0 / sqrt(n);
    for (size_t i = 0; i < m->rows; ++i) {
        for (size_t j = 0; j < m->cols; ++j) {
            m->entries[i * m->cols + j] = uniform_distribution(min, max);
        }
    }
}

void matrix_apply(_activation_func func, Matrix *m) {
    for (size_t i = 0; i < m->rows; ++i) {
        for (size_t j = 0; j < m->cols; ++j) {
            m->entries[i * m->cols + j] = func(m->entries[i * m->cols + j]);
        }
    }
}

void matrix_transpose(Matrix **m) {
    assert(m != NULL);
    assert(*m != NULL);
    Matrix *new = matrix_create((*m)->cols, (*m)->rows);
    for (size_t i = 0; i < (*m)->rows; ++i) {
        for (size_t j = 0; j < (*m)->cols; ++j) {
            new->entries[j * new->cols + i] = (*m)->entries[i * (*m)->cols + j];
        }
    }
    matrix_free(*m);
    *m = new;
}

void matrix_dot(Matrix *dst, Matrix *a, Matrix *b) {
    assert(dst != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->cols == b->rows);
    assert(dst->rows == a->rows);
    assert(dst->cols == b->cols);
    size_t n = a->cols;

    for (size_t i = 0; i < dst->rows; ++i) {
        for (size_t j = 0; j < dst->cols; ++j) {
            float s = 0;
            for (size_t k = 0; k < n; ++k) {
                s += a->entries[i * a->cols + k] * b->entries[k * b->cols + j];
            }
            dst->entries[i * dst->cols + j] = s;
        }
    }
}

void matrix_add(Matrix *dst, Matrix *m) {
    assert(dst != NULL);
    assert(m != NULL);
    assert(dst->rows == m->rows);
    assert(dst->cols == m->cols);
    size_t n = m->cols;

    for (size_t i = 0; i < dst->rows; ++i) {
        for (size_t j = 0; j < dst->cols; ++j) {
            dst->entries[i * dst->cols + j] += m->entries[i * m->cols + j];
        }
    }
}

void matrix_scale(Matrix *dst, float n) {
    assert(dst != NULL);
    for (size_t i = 0; i < dst->rows; ++i) {
        for (size_t j = 0; j < dst->cols; ++j) {
            dst->entries[i * dst->cols + j] *= n;
        }
    }
}

void matrix_subtract(Matrix *dst, Matrix *m) {
    assert(dst != NULL);
    assert(m != NULL);
    assert(dst->rows == m->rows);
    assert(dst->cols == m->cols);

    for (size_t i = 0; i < dst->rows; ++i) {
        for (size_t j = 0; j < dst->cols; ++j) {
            dst->entries[i * dst->cols + j] -= m->entries[i * m->cols + j];
        }
    }
}

void matrix_multiply(Matrix *dst, Matrix *a, Matrix *b) {
    assert(dst != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    assert(dst->rows == a->rows);
    assert(dst->cols == a->cols);

    for (size_t i = 0; i < dst->rows; ++i) {
        for (size_t j = 0; j < dst->cols; ++j) {
            float xa = a->entries[i * a->cols + j];
            float xb = b->entries[i * b->cols + j];
            dst->entries[i * dst->cols + j] = xa * xb;
        }
    }
}
