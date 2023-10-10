#include "nn.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_LEN(arr) (sizeof((arr)) / sizeof((arr)[0]))

int main() {
    srand(time(0));
    rand();

    Matrix *train_input = matrix_create(4, 2);
    train_input->entries[0] = 0;
    train_input->entries[1] = 0;
    train_input->entries[2] = 0;
    train_input->entries[3] = 1;
    train_input->entries[4] = 1;
    train_input->entries[5] = 0;
    train_input->entries[6] = 1;
    train_input->entries[7] = 1;
 
    Matrix *train_output = matrix_create(4, 1);
    train_output->entries[0] = 0;
    train_output->entries[1] = 1;
    train_output->entries[2] = 1;
    train_output->entries[3] = 0;

    const size_t EPOCHS = 100000;
    //const size_t EPOCHS = 50000;
    const float LEARNING_RATE = 0.1f;
    size_t arch[] = { 2, 2, 1 };

    NN *nn = nn_create(arch, ARRAY_LEN(arch));
    nn_randomize(nn);
    nn_print(nn);

    printf("loss = %f\n", nn_loss(nn, train_input, train_output));

    for (size_t i = 0; i < EPOCHS; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            Matrix *x = matrix_row(train_input, j);
            Matrix *y = matrix_row(train_output, j);

            matrix_copy(nn->activations[0], x);
            nn_forward(nn);

            nn_backward(nn, LEARNING_RATE, y);

            matrix_free(x);
            matrix_free(y);
        }
    }

    printf("loss = %f\n", nn_loss(nn, train_input, train_output));

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            nn->activations[0]->entries[0] = i;
            nn->activations[0]->entries[1] = j;
            nn_forward(nn);
            float y = nn->activations[nn->layer_count]->entries[0];
            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }

    return 0;
}

