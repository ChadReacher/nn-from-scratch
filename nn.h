#ifndef NN_H
#define NN_H

#include "matrix.h"

typedef struct {
    size_t layer_count;
    Matrix **weights;
    Matrix **biases;
    Matrix **zs;
    Matrix **activations;
} NN;

NN *nn_create(size_t *arch, size_t arch_count);
void nn_randomize(NN *n);
void nn_forward(NN *nn);
void nn_backward(NN *nn, float learning_rate, Matrix *Y);
float nn_loss(NN *nn, Matrix *train_input, Matrix *train_output);
void nn_print(NN *nn);

#endif
