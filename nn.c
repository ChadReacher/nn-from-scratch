#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

void mse_prime(Matrix *dst, Matrix *Yhat, Matrix *Y) {
    matrix_copy(dst, Yhat);
    matrix_subtract(dst, Y);
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

// User should free the memory
Matrix *sigmoid_prime(Matrix *m) {
    Matrix *ones = matrix_create(m->rows, m->cols);
    matrix_fill(ones, 1);
    matrix_subtract(ones, m);
    Matrix *temp2 = matrix_create(m->rows, m->cols);
    matrix_multiply(temp2, m, ones);

    matrix_free(ones);
    return temp2;
}

NN *nn_create(size_t *arch, size_t arch_count) {
    NN *nn = malloc(sizeof(NN));
    assert(nn != NULL);
    nn->layer_count = arch_count - 1; // excluding the inputs

    nn->weights = malloc(sizeof(Matrix *) * nn->layer_count);
    assert(nn->weights != NULL);
    nn->biases = malloc(sizeof(Matrix *) * nn->layer_count);
    assert(nn->biases != NULL);
    nn->zs = malloc(sizeof(Matrix *) * (nn->layer_count));
    assert(nn->zs != NULL);
    nn->activations = malloc(sizeof(Matrix *) * (nn->layer_count + 1));
    assert(nn->activations != NULL);

    nn->activations[0] = matrix_create(1, arch[0]);
    matrix_fill(nn->activations[0], 0);
    for (size_t i = 1; i < arch_count; ++i) {
        nn->weights[i - 1] = matrix_create(nn->activations[i - 1]->cols, arch[i]);
        matrix_fill(nn->weights[i - 1], 0);

        nn->biases[i - 1] = matrix_create(1, arch[i]);
        matrix_fill(nn->biases[i - 1], 0);

        nn->zs[i - 1] = matrix_create(1, arch[i]);
        matrix_fill(nn->zs[i - 1], 0);

        nn->activations[i] = matrix_create(1, arch[i]);
        matrix_fill(nn->activations[i], 0);
    }

    return nn;
}

void nn_forward(NN *nn) {
    for (size_t i = 1; i <= nn->layer_count; ++i) {
        matrix_dot(nn->activations[i], nn->activations[i - 1], nn->weights[i - 1]);
        matrix_add(nn->activations[i], nn->biases[i - 1]);
        matrix_apply(sigmoidf, nn->activations[i]);
    }
}

void nn_backward(NN *nn, float learning_rate, Matrix *Y) {
    Matrix *grad = matrix_create(nn->activations[nn->layer_count]->rows, nn->activations[nn->layer_count]->cols);
    mse_prime(grad, nn->activations[nn->layer_count], Y);

    for (size_t layer = nn->layer_count; layer > 0; --layer) {
        Matrix *temp;
        // Compute the new temporary gradient
        if (layer == nn->layer_count) {
            Matrix *temp2 = sigmoid_prime(nn->activations[layer]);
            temp = matrix_create(grad->rows, grad->cols);
            matrix_multiply(temp, grad, temp2);

            matrix_free(temp2);
        } else {
            Matrix *W2_T = matrix_create(nn->weights[layer]->rows, nn->weights[layer]->cols);
            matrix_copy(W2_T, nn->weights[layer]);
            matrix_transpose(&W2_T);
            temp = matrix_create(grad->rows, W2_T->cols);
            matrix_dot(temp, grad, W2_T);
            
            Matrix *temp2 = sigmoid_prime(nn->activations[layer]);
            matrix_multiply(temp, temp, temp2);

            matrix_free(temp2);
            matrix_free(W2_T);
        }

        // Update the temporary gradient
        matrix_free(grad);
        grad = matrix_create(temp->rows, temp->cols);
        matrix_copy(grad, temp);
        matrix_free(temp);

        // Update weights
        Matrix *Ol_T = matrix_create(nn->activations[layer - 1]->rows, nn->activations[layer - 1]->cols);
        matrix_copy(Ol_T, nn->activations[layer - 1]);
        matrix_transpose(&Ol_T);
        Matrix *dldwl = matrix_create(Ol_T->rows, grad->cols);
        matrix_dot(dldwl, Ol_T, grad);
        matrix_scale(dldwl, learning_rate);
        matrix_subtract(nn->weights[layer - 1], dldwl);

        // Update the biases
        Matrix *dldbl = matrix_create(grad->rows, grad->cols);
        matrix_copy(dldbl, grad);
        matrix_scale(dldbl, learning_rate);
        matrix_subtract(nn->biases[layer - 1], dldbl);

        matrix_free(dldbl);
        matrix_free(dldwl);
        matrix_free(Ol_T);
    }
    matrix_free(grad);
}

float nn_loss(NN *nn, Matrix *train_input, Matrix *train_output) {
    assert(train_input->rows == train_output->rows);
    assert(train_output->cols == nn->activations[nn->layer_count]->cols);
    size_t n = train_input->rows;
    
    float cost = 0;
    Matrix *train_set = NULL;
    for (size_t i = 0; i < n; ++i) {
        Matrix *X = matrix_row(train_input, i);
        Matrix *Y = matrix_row(train_output, i);

        matrix_copy(nn->activations[0], X);
        nn_forward(nn);

        // Softmax
        float exps = 0;
        for (size_t i = 0; i < nn->activations[nn->layer_count]->cols; ++i) {
            exps += expf(nn->activations[nn->layer_count]->entries[i]);
        }
        for (size_t i = 0; i < nn->activations[nn->layer_count]->cols; ++i) {
            float x = nn->activations[nn->layer_count]->entries[i];
            nn->activations[nn->layer_count]->entries[i] = expf(x) / exps;
        }

        float actual, expected;
        Matrix *nn_output_matrix = nn->activations[nn->layer_count];
        for (size_t j = 0; j < train_output->cols; ++j) {
            actual = nn_output_matrix->entries[j];
            expected = Y->entries[j]; 
            float d = expected - actual;
            cost += d * d;
        }
        matrix_free(X);
        matrix_free(Y);
    }

    return cost / n;
}

void nn_randomize(NN *nn) {
    for (size_t i = 0; i < nn->layer_count; ++i) {
        matrix_randomize(nn->weights[i], nn->weights[i]->cols);
        matrix_randomize(nn->biases[i], nn->biases[i]->cols);
    }
}

void nn_print(NN *nn) {
    for (size_t i = 0; i < nn->layer_count; ++i) {
        printf("w[%d] = ", i);
        matrix_print(nn->weights[i]);
        printf("b[%d] = ", i);
        matrix_print(nn->biases[i]);
    }
}
