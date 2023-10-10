#include "nn.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define MAXCHAR 10000
#define ARRAY_LEN(arr) (sizeof((arr)) / sizeof((arr)[0]))

typedef struct Img {
    Matrix *data;
    size_t label;
} Img;

int main(void) {
    srand(time(0));
    rand();

    char row[MAXCHAR] = { 0 };
    const size_t number_of_train_imgs = 3000;
    const size_t number_of_test_imgs = 1000;
    const size_t EPOCHS = 100;
    const float LEARNING_RATE = 0.1f;
    const char *train_set_path = "data/mnist_train.csv";
    const char *test_set_path = "data/mnist_test.csv";
    size_t arch[] = { 784, 300, 10 };

    Img **imgs = malloc(number_of_train_imgs * sizeof(Img *));
    FILE *fp = fopen(train_set_path, "r");
    fgets(row, MAXCHAR, fp);
    size_t i = 0;
    while (feof(fp) != 1 && i < number_of_train_imgs) {
        imgs[i] = malloc(sizeof(Img));

        size_t j = 0;
        fgets(row, MAXCHAR, fp);
        char *token = strtok(row, ",");
        imgs[i]->data = matrix_create(1, 28 * 28);
        while (token != NULL) {
            if (j == 0) {
                imgs[i]->label = atoi(token);
            } else {
                size_t x = (j - 1) / 28;
                size_t y = (j - 1) % 28;
                imgs[i]->data->entries[x * 28 + y] = atoi(token) / 256.0;
            }
            token = strtok(NULL, ",");
            ++j;
        }
        ++i;
    }
    fclose(fp);
    printf("Loaded the data\n");

    NN *nn = nn_create(arch, ARRAY_LEN(arch));
    nn_randomize(nn);

    Matrix *big_input = matrix_create(number_of_train_imgs, 784);
    Matrix *big_output = matrix_create(number_of_train_imgs, 10);
    for (size_t k = 0; k < number_of_train_imgs; ++k) {
        memcpy(big_input->entries + k * big_input->cols, imgs[k]->data->entries, 784 * sizeof(float));
        memset(big_output->entries + k * big_output->cols, 0, 10 * sizeof(float));
        big_output->entries[k * big_output->cols + imgs[k]->label] = 1.0f;
    }

    Matrix *train_output = matrix_create(1, 10);
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        for (size_t img_idx = 0; img_idx < number_of_train_imgs; ++img_idx) {
            Img *img = imgs[img_idx];
            Matrix *img_matrix = img->data;

            memset(train_output->entries, 0, 10 * sizeof(float));
            train_output->entries[img->label] = 1.0f;

            nn->activations[0] = img_matrix;

            nn_forward(nn);

            nn_backward(nn, LEARNING_RATE, train_output);
        }
        printf("epoch #: %d\n", epoch);
    }
    matrix_free(train_output);

    fp = fopen(test_set_path, "r");
    Img **test_imgs = malloc(number_of_test_imgs * sizeof(Img *));
    fgets(row, MAXCHAR, fp);
    i = 0;
    while (feof(fp) != 1 && i < number_of_test_imgs) {
        test_imgs[i] = malloc(sizeof(Img));

        size_t j = 0;
        fgets(row, MAXCHAR, fp);
        char *token = strtok(row, ",");
        test_imgs[i]->data = matrix_create(1, 28 * 28);
        while (token != NULL) {
            if (j == 0) {
                test_imgs[i]->label = atoi(token);
            } else {
                size_t x = (j - 1) / 28;
                size_t y = (j - 1) % 28;
                test_imgs[i]->data->entries[x * 28 + y] = atoi(token) / 256.0;
            }
            token = strtok(NULL, ",");
            ++j;
        }
        ++i;
    }
    fclose(fp);

    float ncorrect = 0;
    for (size_t img_idx = 0; img_idx < number_of_test_imgs; ++img_idx) {
        Img *img = test_imgs[img_idx];
        nn->activations[0] = img->data;
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

        float actual = 0.0f;
        size_t n = 0;
        for (size_t i = 0; i < 10; ++i) {
            if (nn->activations[nn->layer_count]->entries[i] > actual) {
                actual = nn->activations[nn->layer_count]->entries[i];
                n = i;
            }
        }
        
        if (n == img->label) {
            ++ncorrect;
        }
        printf("actual: %d, expected: %d\n", n, img->label);
    }    
    printf("accuracy: %f", ncorrect / number_of_test_imgs);
    
    return 0;
}
