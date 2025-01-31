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
    NeuralNet model;
    size_t layers[] = {2, 2, 1};
    nl_define_layers(&model, NL_ARRAY_LEN(layers), layers);
    Mat x = nl_mat_alloc(2, 1);
    Mat y = nl_mat_alloc(1, 1);

    for (size_t i = 0; i < NL_ARRAY_LEN(layers) - 1; ++i) {
        nl_mat_print(model.ws[i]);
        nl_mat_print(model.bs[i]);
    }

    printf("----------\n");
    float lr = 1e-3;
    size_t epoch = 2000 * 1000;
    for (size_t e = 0; e < epoch; ++e) {
        for (size_t i = 0; i < TRAIN_SIZE; ++i) {
            x.items[0] = train[i][0];
            x.items[1] = train[i][1];
            y.items[0] = train[i][2];
            nl_model_train(model, x, y, lr, SIGMOID, MSE);
            // nl_model_train(model, x, y, lr, RELU, MSE);
        }
    }

    // Predict
    Mat px = nl_mat_alloc(2, 1);
    Mat py = nl_mat_alloc(1, 1);
    for (size_t i = 0; i < 4; ++i) {
        px.items[0] = train[i][0];
        px.items[1] = train[i][1];
        nl_model_predict(model, px, py, SIGMOID);
        // nl_model_predict(model, px, py, RELU);
        printf("%f , expect: %f\n", py.items[0], train[i][2]);
    }

    printf("===== Final\n");
    for (size_t i = 0; i < NL_ARRAY_LEN(layers) - 1; ++i) {
        nl_mat_print(model.ws[i]);
        nl_mat_print(model.bs[i]);
    }

    nl_model_free(model);
}
