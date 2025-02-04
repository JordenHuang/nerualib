#include <stdio.h>

#define NERUALIB_IMPLEMENTATION
#include "../neuralib.h"

#define TRAIN_SIZE (sizeof(train)/sizeof(train[0]))

// XOR gate
float train[][3] = {
    { 0, 0, 0 },
    { 0, 1, 1 },
    { 1, 0, 1 },
    { 1, 1, 0 },
};

int main(void)
{
    nl_rand_init(0, 0);

    NeuralNet model;
    size_t layers[] = {2, 2, 1};
    // Activation_type acts[] = {SIGMOID, RELU};
    Activation_type acts[] = {RELU, SIGMOID};
    nl_define_layers(&model, NL_ARRAY_LEN(layers), layers, acts, MSE);

    for (size_t i = 0; i < NL_ARRAY_LEN(layers) - 1; ++i) {
        nl_mat_print(model.ws[i]);
        nl_mat_print(model.bs[i]);
    }

    // Prepare training data
    Mat new_x = nl_mat_alloc(2, TRAIN_SIZE);
    Mat new_y = nl_mat_alloc(1, TRAIN_SIZE);
    for (size_t i = 0; i < TRAIN_SIZE; ++i) {
        for (size_t a = 0; a < new_x.rows; ++a) {
            NL_MAT_AT(new_x, a, i) = train[i][a];
        }
        for (size_t a = 0; a < new_y.rows; ++a) {
            NL_MAT_AT(new_y, a, i) = train[i][a + new_x.rows];
        }
    }
    nl_mat_print(new_x);
    nl_mat_print(new_y);

    printf("===== Train\n");
    size_t epochs = 100 * 100;
    float lr = 5e-1;
    nl_model_train(model, new_x, new_y, lr, epochs, 1, false);


    // Predict
    printf("===== Predict\n");
    Mat px = nl_mat_alloc(2, 1);
    Mat py = nl_mat_alloc(1, 1);
    for (size_t i = 0; i < 4; ++i) {
        px.items[0] = train[i][0];
        px.items[1] = train[i][1];
        nl_model_predict(model, px, py);
        printf("%f , expect: %f\n", py.items[0], train[i][2]);
    }

    printf("===== Final\n");
    for (size_t i = 0; i < NL_ARRAY_LEN(layers) - 1; ++i) {
        nl_mat_print(model.ws[i]);
        nl_mat_print(model.bs[i]);
    }

    nl_mat_free(new_x);
    nl_mat_free(new_y);
    nl_mat_free(px);
    nl_mat_free(py);
    nl_model_free(model);
}
